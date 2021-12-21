import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


class CustomDataset(Dataset):
    def __init__(self, df_file_name, max_samples=-1):
        df = pd.read_csv(df_file_name)
        if (max_samples != -1):
            df = df.head(max_samples)
        self.labels = list(df["label"])
        del df["label"]
        self.features = np.array(df).tolist()

    def __getitem__(self, index):
        return {
            "features": torch.tensor(self.features[index]),
            "labels": torch.tensor(self.labels[index]),
        }

    def __len__(self):
        return len(self.labels)


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
        self.fc1 = torch.nn.Linear(model_arguments.number_of_features, self.model_arguments.fc1_size)
        self.fc2 = torch.nn.Linear(self.model_arguments.fc1_size, self.model_arguments.num_labels)

        self.optimizer = Adam
        self.save_hyperparameters("training_arguments")
        self.save_hyperparameters("model_arguments")

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        return x

    def _step(self, batch):
        outputs = self(batch["features"])
        logits = F.log_softmax(outputs, dim=1)
        softmax_logits = F.softmax(outputs, dim=1)
        loss = F.nll_loss(logits, batch["labels"])
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
        return self.optimizer(self.parameters(), lr=self.other_arguments.learning_rate)

    def train_dataloader(self):
        train_dataset = CustomDataset(
            df_file_name=self.other_arguments.TRAIN_FILE,
            max_samples=self.other_arguments.max_train_samples,
        )
        dataloader = DataLoader(
            train_dataset,
            self.other_arguments.train_batch_size,
            drop_last=False, shuffle=True,
            num_workers=self.training_arguments.num_workers)

        return dataloader

    def val_dataloader(self):
        val_dataset = CustomDataset(
            df_file_name=self.other_arguments.DEV_FILE,
        )

        return DataLoader(val_dataset,
                          batch_size=self.other_arguments.eval_batch_size,
                          num_workers=self.training_arguments.num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training arguments
    training_arguments = parser.add_argument_group('training_arguments')
    training_arguments.add_argument("--opt_level", default="O1")
    training_arguments.add_argument('--max_grad_norm', type=float, default=1.0)
    training_arguments.add_argument("--fp_16", default=False, action="store_true")
    training_arguments.add_argument("--n_gpu", default=-1, type=int)
    training_arguments.add_argument("--num_workers", default=8, type=int)
    training_arguments.add_argument("--distributed_backend", default=None)

    # Model arguments
    model_arguments = parser.add_argument_group('model_arguments')
    model_arguments.add_argument("--num_labels", type=int)
    model_arguments.add_argument("--number_of_features", type=int)
    model_arguments.add_argument("--fc1_size", type=int)

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

    '''
    args = parser.parse_args(
    " --model_name_or_path roberta-base  --max_input_seq_length 100   --TRAIN_FILE sst2_train.csv --output_dir ./ --DEV_FILE sst2_dev.csv --train_batch_size 32 --eval_batch_size 32 --max_train_samples 10000 --num_train_epochs 5 --gradient_accumulation_steps 1 --save_top_k 0 --learning_rate 5e-6 --write_dev_predictions".split()
    )

    '''
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
    model = ClassificationModel(training_arguments=training_arguments,
                                model_arguments=model_arguments,
                                other_arguments=other_arguments)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=other_arguments.output_dir,
        monitor="val_acc",
        save_top_k=other_arguments.save_top_k,
        save_last=other_arguments.save_last,
        mode='max'
    )

    train_params = dict(
        accumulate_grad_batches=other_arguments.gradient_accumulation_steps,
        gpus=training_arguments.n_gpu,
        deterministic=True,
        max_epochs=other_arguments.num_train_epochs,
        precision=16 if training_arguments.fp_16 else 32,
        amp_level=training_arguments.opt_level,
        gradient_clip_val=training_arguments.max_grad_norm,
        callbacks=[checkpoint_callback],
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
