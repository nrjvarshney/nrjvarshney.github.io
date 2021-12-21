import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    T5ForConditionalGeneration,
    T5Tokenizer
)


class CustomDataset(Dataset):
    def __init__(self, tokenizer, df_file_name, input_max_len=512, output_max_len=512, max_samples=-1):
        self.tokenizer = tokenizer
        self.df = pd.read_csv(df_file_name)
        if (max_samples != -1):
            self.df = self.df.head(max_samples)
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len

    def __getitem__(self, index):
        sentence1 = str(self.df.iloc[index]["sentence1"]).strip()
        sentence2 = str(self.df.iloc[index]["sentence2"]).strip()

        source_encoding = self._get_encoding(
            sentence1=sentence1,
            sentence2=None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.input_max_len,
            padding='max_length',
        )

        target_encoding = self._get_encoding(
            sentence1=sentence2,
            sentence2=None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.output_max_len,
            padding='max_length',
        )

        # source_ids, src_mask = self._get_ids_n_mask(text_a=sentence1, text_b=None, max_length=self.input_max_len)
        # target_ids, target_mask = self._get_ids_n_mask(text_a=sentence2, text_b=None, max_length=self.output_max_len)

        return {
            "source_ids": source_encoding["input_ids"],
            "source_mask": source_encoding["attention_mask"],
            "target_ids": target_encoding["input_ids"],
            "target_mask": target_encoding["attention_mask"],
        }

    def _get_encoding(self, sentence1, sentence2, add_special_tokens=False, truncation=True, max_length=-1,
                      padding=None):
        encoded_input = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        if "num_truncated_tokens" in encoded_input and encoded_input["num_truncated_tokens"] > 0:
            # print("Attention! you are cropping tokens")
            pass
        input_ids = encoded_input["input_ids"].squeeze(0)
        attention_mask = encoded_input["attention_mask"].squeeze(0) if "attention_mask" in encoded_input else None
        token_type_ids = encoded_input["token_type_ids"].squeeze(0) if "token_type_ids" in encoded_input else None
        data_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if (token_type_ids != None):
            data_input["token_type_ids"] = token_type_ids
        return data_input

    def __len__(self):
        return self.df.shape[0]


class T5FineTuner(pl.LightningModule):
    def __init__(self, training_arguments, model_arguments, other_arguments):
        super(T5FineTuner, self).__init__()

        self.training_arguments = training_arguments
        self.model_arguments = model_arguments
        self.other_arguments = other_arguments
        self.tokenizer = T5Tokenizer.from_pretrained(model_arguments.model_name_or_path)

        self.model = T5ForConditionalGeneration.from_pretrained(model_arguments.model_name_or_path)
        self.save_hyperparameters("training_arguments")
        self.save_hyperparameters("model_arguments")

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.cat([x['loss'].view(-1) for x in outputs]).mean()

        print("--------------------")
        print("Train avg_loss: ", avg_loss)
        print("--------------------")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)

        outs = self.model.generate(input_ids=batch['source_ids'],
                                    attention_mask=batch['source_mask'],
                                    max_length=self.model_arguments.max_output_seq_length)
        preds_cleaned = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for ids in outs]

        self.log('val_loss', loss, on_epoch=True)
        return {
            "val_loss": loss,
            "predictions": preds_cleaned,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x['val_loss'].view(-1) for x in outputs]).mean()

        all_predictions = []

        for x in outputs:
            all_predictions.extend(x["predictions"])

        print("--------------------")
        print("Validation avg_loss: ", avg_loss)

        pred_df = pd.DataFrame({
            "prediction": all_predictions,
        })

        if (self.other_arguments.write_dev_predictions):
            output_path = self.other_arguments.output_dir + "epoch_" + str(
                self.trainer.current_epoch) + "_" + self.other_arguments.predictions_file
            print(f"Writing predictions for {self.other_arguments.DEV_FILE} to {output_path}")
            pred_df.to_csv(output_path, index=False)
        print("--------------------")

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_arguments.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.other_arguments.learning_rate,
                          eps=self.training_arguments.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        train_dataset = CustomDataset(
            tokenizer=self.tokenizer,
            df_file_name=self.other_arguments.TRAIN_FILE,
            input_max_len=self.model_arguments.max_input_seq_length,
            output_max_len=self.model_arguments.max_output_seq_length,
            max_samples=self.other_arguments.max_train_samples,
        )
        dataloader = DataLoader(
            train_dataset,
            self.other_arguments.train_batch_size,
            drop_last=True, shuffle=True,
            num_workers=self.training_arguments.num_workers)

        t_total = (
                (len(dataloader.dataset) // (
                        self.other_arguments.train_batch_size * max(1, self.training_arguments.n_gpu)))
                // self.other_arguments.gradient_accumulation_steps
                * float(self.other_arguments.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.training_arguments.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = CustomDataset(
            tokenizer=self.tokenizer,
            df_file_name=self.other_arguments.DEV_FILE,
            input_max_len=self.model_arguments.max_input_seq_length,
            output_max_len=self.model_arguments.max_output_seq_length,
        )

        return DataLoader(val_dataset,
                          batch_size=self.other_arguments.eval_batch_size,
                          num_workers=self.training_arguments.num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training arguments
    training_arguments = parser.add_argument_group('training_arguments')
    training_arguments.add_argument("--opt_level", default="O1")
    training_arguments.add_argument("--warmup_steps", default=0, type=int)
    training_arguments.add_argument('--weight_decay', type=float, default=0.0)
    training_arguments.add_argument('--adam_epsilon', type=float, default=1e-8)
    training_arguments.add_argument('--max_grad_norm', type=float, default=1.0)
    training_arguments.add_argument("--early_stop_callback", default=False, action="store_true")
    training_arguments.add_argument("--fp_16", default=False, action="store_true")
    training_arguments.add_argument("--n_gpu", default=-1, type=int)
    training_arguments.add_argument("--num_workers", default=8, type=int)
    training_arguments.add_argument("--distributed_backend", default=None)

    # Model arguments
    model_arguments = parser.add_argument_group('model_arguments')
    model_arguments.add_argument("--model_name_or_path", default=None)
    model_arguments.add_argument("--max_input_seq_length", default=512, type=int)
    model_arguments.add_argument("--max_output_seq_length", default=512, type=int)

    # Other arguments
    other_arguments = parser.add_argument_group('other_arguments')
    other_arguments.add_argument("--output_dir", default="./")
    other_arguments.add_argument("--predictions_file", default="predictions.csv")
    other_arguments.add_argument("--TRAIN_FILE", default=None)
    other_arguments.add_argument("--DEV_FILE", default=None)
    other_arguments.add_argument("--train_batch_size", default=2, type=int)
    other_arguments.add_argument("--eval_batch_size", default=2, type=int)
    other_arguments.add_argument("--max_train_samples", default=-1, type=int)
    other_arguments.add_argument("--num_train_epochs", default=2, type=int)
    other_arguments.add_argument("--gradient_accumulation_steps", default=1, type=int)
    other_arguments.add_argument("--seed", default=42, type=int)
    other_arguments.add_argument("--save_top_k", default=-1, type=int)
    other_arguments.add_argument("--save_last", default=False, action="store_true")
    other_arguments.add_argument("--write_dev_predictions", default=False, action="store_true")
    other_arguments.add_argument('--learning_rate', type=float, default=3e-4)

    other_arguments.add_argument("--do_fast_dev_run", default=False, action="store_true")
    other_arguments.add_argument("--limit_train_batches", default=-1, type=int)
    other_arguments.add_argument("--limit_val_batches", default=-1, type=int)

    args = parser.parse_args()

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if (group.title == "training_arguments"):
            training_arguments = argparse.Namespace(**group_dict)
        elif (group.title == "model_arguments"):
            model_arguments = argparse.Namespace(**group_dict)
        elif (group.title == "other_arguments"):
            other_arguments = argparse.Namespace(**group_dict)

    print("Training arguments", training_arguments)
    print("--------------------")
    print("Model arguments", model_arguments)
    print("--------------------")
    print("Other arguments", other_arguments)
    print("--------------------")

    pl.seed_everything(other_arguments.seed)
    model = T5FineTuner(training_arguments=training_arguments,
                        model_arguments=model_arguments,
                        other_arguments=other_arguments
            )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=other_arguments.output_dir,
        monitor="val_loss",
        save_top_k=other_arguments.save_top_k,
        save_last=other_arguments.save_last,
        mode='min'
    )

    train_params = dict(
        accumulate_grad_batches=other_arguments.gradient_accumulation_steps,
        gpus=training_arguments.n_gpu,
        deterministic=True,
        max_epochs=other_arguments.num_train_epochs,
        precision=16 if training_arguments.fp_16 else 32,
        amp_level=training_arguments.opt_level,
        gradient_clip_val=training_arguments.max_grad_norm,
        callbacks=checkpoint_callback,
        fast_dev_run=other_arguments.do_fast_dev_run,
    )

    if (other_arguments.limit_train_batches != -1):
        train_params["limit_train_batches"] = other_arguments.limit_train_batches

    if (other_arguments.limit_val_batches != -1):
        train_params["limit_val_batches"] = other_arguments.limit_val_batches

    if (training_arguments.distributed_backend != None):
        train_params["distributed_backend"] = training_arguments.distributed_backend

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
