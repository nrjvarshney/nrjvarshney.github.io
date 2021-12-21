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
import tqdm

class CustomDatasetEval(Dataset):
    def __init__(self, tokenizer, df, df_file_name, input_max_len=512, output_max_len=512, max_samples=-1):
        self.tokenizer = tokenizer

        if(df is not None):
            self.df = df
        else:
            self.df = pd.read_csv(df_file_name)
        if (max_samples != -1):
            self.df = self.df.head(max_samples)
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len

    def __getitem__(self, index):
        sentence1 = str(self.df.iloc[index]["sentence1"]).strip()

        source_encoding = self._get_encoding(
            sentence1=sentence1,
            sentence2=None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.input_max_len,
            padding='max_length',
        )

        return {
            "source_ids": source_encoding["input_ids"],
            "source_mask": source_encoding["attention_mask"],
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


def evaluate(model, df, other_arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.model.eval()
    outputs = []
    test_dataset = CustomDatasetEval(
        tokenizer=model.tokenizer,
        df=df,
        df_file_name=other_arguments.DEV_FILE,
        input_max_len=model.model_arguments.max_input_seq_length,
        output_max_len=model.model_arguments.max_output_seq_length
    )

    test_dataloader = DataLoader(
        test_dataset,
        other_arguments.eval_batch_size,
        num_workers=model.training_arguments.num_workers)

    for batch in tqdm.tqdm(test_dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)

        outs = model.model.generate(input_ids=batch['source_ids'],
                                    attention_mask=batch['source_mask'],
                                    max_length=model.model_arguments.max_output_seq_length)
        preds_cleaned = [model.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for ids in outs]
        outputs.extend(preds_cleaned)
    df = pd.DataFrame({
        "prediction": outputs,
    })
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Other arguments
    other_arguments = parser.add_argument_group('other_arguments')
    other_arguments.add_argument("--checkpoint_path", default=None)
    other_arguments.add_argument("--output_dir", default="./")
    other_arguments.add_argument("--predictions_file", default="predictions.csv")
    other_arguments.add_argument("--DEV_FILE", default=None)
    other_arguments.add_argument("--eval_batch_size", default=2, type=int)
    other_arguments.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if (group.title == "other_arguments"):
            other_arguments = argparse.Namespace(**group_dict)

    print("Other arguments", other_arguments)
    print("--------------------")

    pl.seed_everything(other_arguments.seed)
    model = T5FineTuner.load_from_checkpoint(checkpoint_path=other_arguments.checkpoint_path,
                                            other_arguments=other_arguments
            )
    eval_df = pd.DataFrame({
        "sentence1": ["A man is sitting with a boy."],
    })
    if(other_arguments.DEV_FILE != None):
        pred_df = evaluate(model=model, df=None, other_arguments=other_arguments)
    else:
        pred_df = evaluate(model=model, df=eval_df, other_arguments=other_arguments)
    print(pred_df)
