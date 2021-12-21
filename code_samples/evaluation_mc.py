import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import pandas as pd
import argparse
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
import tqdm


class NLIDataset(Dataset):
    def __init__(self, tokenizer, df_file_name, input_max_len=512, max_samples=-1):
        self.tokenizer = tokenizer
        self.df = pd.read_csv(df_file_name)
        if (max_samples != -1):
            self.df = self.df.head(max_samples)
        self.input_max_len = input_max_len

    def __getitem__(self, index):
        sentence1 = str(self.df.iloc[index]["sentence1"])

        if ('sentence2' in self.df.columns):
            sentence2 = str(self.df.iloc[index]["sentence2"])
        else:
            sentence2 = None

        label = self.df.iloc[index]["label"]

        instance_encoding = self._get_encoding(
            sentence1=sentence1,
            sentence2=sentence2,
            add_special_tokens=True,
            truncation=True,
            max_length=self.input_max_len,
            padding='max_length',
        )
        return {
            "input_ids": instance_encoding["input_ids"],
            "attention_mask": instance_encoding["attention_mask"],
            "labels": torch.tensor(label),
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


def compute_accuracy(logits, labels):
    predicted_label = logits.max(dim=1)[1]
    acc = (predicted_label == labels).float().mean()
    return acc, predicted_label


class ClassificationModel(pl.LightningModule):
    def __init__(self, training_arguments, model_arguments, other_arguments):
        super(ClassificationModel, self).__init__()

        self.training_arguments = training_arguments
        self.model_arguments = model_arguments
        self.other_arguments = other_arguments
        self.tokenizer = AutoTokenizer.from_pretrained(model_arguments.model_name_or_path)
        config = AutoConfig.from_pretrained(model_arguments.model_name_or_path,
                                            num_labels=model_arguments.num_labels,
                                            hidden_dropout_prob=model_arguments.hidden_dropout_prob)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_arguments.model_name_or_path, config=config)
        self.save_hyperparameters("training_arguments")
        self.save_hyperparameters("model_arguments")

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def forward(self,
                input_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    def _step(self, batch):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        softmax_logits = F.softmax(logits, dim=1)
        return loss, softmax_logits

    def training_step(self, batch, batch_idx):
        loss, logits = self._step(batch)
        acc, predicted_label = compute_accuracy(logits, batch["labels"])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.cat([x['loss'].view(-1) for x in outputs]).mean()
        avg_acc = torch.cat([x['acc'].view(-1) for x in outputs]).mean()

        print("--------------------")
        print("Train avg_loss: ", avg_loss)
        print("Train avg_acc: ", avg_acc)
        print("--------------------")

    def validation_step(self, batch, batch_idx):
        loss, logits = self._step(batch)
        logits = logits.squeeze(1)
        acc, predicted_label = compute_accuracy(logits, batch["labels"])
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return {
            "val_loss": loss,
            "val_acc": acc,
            "softmax_logits": logits.tolist(),
            "labels": batch["labels"].tolist(),
            "predictions": predicted_label.tolist(),
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x['val_loss'].view(-1) for x in outputs]).mean()
        avg_acc = torch.cat([x['val_acc'].view(-1) for x in outputs]).mean()

        all_labels = []
        all_predictions = []
        all_softmax_logits = []

        for x in outputs:
            all_predictions += torch.tensor(x["predictions"]).tolist()
            all_softmax_logits += torch.tensor(x["softmax_logits"]).tolist()
            all_labels += torch.tensor(x["labels"]).tolist()

        softmax_logits_df = pd.DataFrame(all_softmax_logits)
        print("--------------------")
        print("Validation avg_loss: ", avg_loss)
        print("Validation avg_acc: ", avg_acc)

        result_df = pd.DataFrame({
            "label": all_labels,
            "prediction": all_predictions,
        })

        result_df = pd.concat([result_df, softmax_logits_df], axis=1)

        if (self.other_arguments.write_dev_predictions):
            output_path = self.other_arguments.output_dir + "epoch_" + str(
                self.trainer.current_epoch) + "_" + self.other_arguments.predictions_file
            print(f"Writing predictions for {self.other_arguments.DEV_FILE} to {output_path}")
            result_df.to_csv(output_path, index=False)
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
        train_dataset = NLIDataset(
            tokenizer=self.tokenizer,
            df_file_name=self.other_arguments.TRAIN_FILE,
            input_max_len=self.model_arguments.max_input_seq_length,
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
        val_dataset = NLIDataset(
            tokenizer=self.tokenizer,
            df_file_name=self.other_arguments.DEV_FILE,
            input_max_len=self.model_arguments.max_input_seq_length,
        )

        return DataLoader(val_dataset,
                          batch_size=self.other_arguments.eval_batch_size,
                          num_workers=self.training_arguments.num_workers)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def evaluate(model, other_arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for pred_idx in range(other_arguments.num_dropout_masks):

        model.eval()
        enable_dropout(model)
        outputs = []
        with torch.no_grad():
            test_dataset = NLIDataset(
                model.tokenizer,
                other_arguments.DEV_FILE,
                model.model_arguments.max_input_seq_length,
                max_samples=-1
            )
            test_dataloader = DataLoader(
                test_dataset,
                other_arguments.eval_batch_size,
                num_workers=model.training_arguments.num_workers)

            for batch in tqdm.tqdm(test_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(device)

                loss, logits = model._step(batch)
                logits = logits.squeeze(1)
                acc, predicted_label = compute_accuracy(logits, batch["labels"])
                outputs.append({
                    "val_loss": loss,
                    "val_acc": acc,
                    "softmax_logits": logits.tolist(),
                    "labels": batch["labels"].tolist(),
                    "predictions": predicted_label.tolist(),
                })

            avg_loss = torch.cat([x['val_loss'].view(-1) for x in outputs]).mean()
            avg_acc = torch.cat([x['val_acc'].view(-1) for x in outputs]).mean()

            all_labels = []
            all_predictions = []
            all_softmax_logits = []

            for x in outputs:
                all_predictions += torch.tensor(x["predictions"]).tolist()
                all_softmax_logits += torch.tensor(x["softmax_logits"]).tolist()
                all_labels += torch.tensor(x["labels"]).tolist()

            softmax_logits_df = pd.DataFrame(all_softmax_logits)
            print("--------------------")
            print("Validation avg_loss: ", avg_loss)
            print("Validation avg_acc: ", avg_acc)

            result_df = pd.DataFrame({
                "label": all_labels,
                "prediction": all_predictions,
            })

            result_df = pd.concat([result_df, softmax_logits_df], axis=1)

            if (other_arguments.write_dev_predictions):
                output_path = other_arguments.output_dir + "dropout_idx_" + str(pred_idx) + "_" + other_arguments.predictions_file
                print(f"Writing predictions for {other_arguments.DEV_FILE} to {output_path}")
                result_df.to_csv(output_path, index=False)
            print("--------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Other arguments
    other_arguments = parser.add_argument_group('other_arguments')
    other_arguments.add_argument("--checkpoint_path", default="./")
    other_arguments.add_argument("--output_dir", default="./")
    other_arguments.add_argument("--predictions_file", default="predictions.csv")
    other_arguments.add_argument("--DEV_FILE", default=None)
    other_arguments.add_argument("--eval_batch_size", default=32, type=int)
    other_arguments.add_argument("--seed", default=42, type=int)
    other_arguments.add_argument("--num_dropout_masks", default=2, type=int)
    other_arguments.add_argument("--write_dev_predictions", default=False, action="store_true")

    '''
    args = parser.parse_args(
    " --model_name_or_path roberta-base  --max_input_seq_length 100   --TRAIN_FILE sst2_train.csv --output_dir ./ --DEV_FILE sst2_dev.csv --train_batch_size 32 --eval_batch_size 32 --max_train_samples 10000 --num_train_epochs 5 --gradient_accumulation_steps 1 --save_top_k 0 --learning_rate 5e-6 --write_dev_predictions".split()
    )
    '''
    args = parser.parse_args()

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if (group.title == "other_arguments"):
            other_arguments = argparse.Namespace(**group_dict)

    print("Other arguments", other_arguments)
    print("--------------------")

    pl.seed_everything(other_arguments.seed)
    model = ClassificationModel.load_from_checkpoint(checkpoint_path=other_arguments.checkpoint_path,
                                                     other_arguments=other_arguments)

    evaluate(model, other_arguments)

