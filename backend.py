# import torch
# print(torch.__version__)
# print(torch.cuda.is_available()) 

from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import Dataset
from datasets import DatasetDict
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, pipeline


import collections
import numpy as np
from transformers import default_data_collator

import torch

# import os
# os.chdir("Users/hanish")

class MLM():
    def __init__(self, data) -> None:
        self.data = data
        self.model_checkpoint = "google-bert/bert-base-multilingual-uncased"
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.chunk_size = 128


    def tokenize_function(self, examples):
        result = self.tokenizer(examples["text"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    
    def group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // self.chunk_size) * self.chunk_size
        result = {
            k: [t[i : i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    def start(self):
        with open(self.data, 'r', encoding='utf-8') as f:
            sent = [line.strip() for line in f]

        train = [{"text": sentence} for sentence in sent]
        train = Dataset.from_list(train)

        self.dataset = DatasetDict({
            "train": train,
            # "validation": val
        })

        tokenized_datasets = self.dataset.map(
            self.tokenize_function, batched=True, remove_columns=["text"]
        )

        
        lm_datasets = tokenized_datasets.map(self.group_texts, batched=True)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

        train_size = 0.9
        test_size = 0.1

        downsampled_dataset = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )
        # downsampled_dataset

        batch_size = 64
        logging_steps = len(downsampled_dataset["train"]) // batch_size
        model_name = self.model_checkpoint.split("/")[-1]

        training_args = TrainingArguments(
            output_dir=f"{model_name}-finetuned",
            # overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
        )

        print("Starrting Training......")

        device = "cuda:0" 

        trainer = Trainer(
            model=self.model.to(device),
            args=training_args,
            train_dataset=downsampled_dataset["train"],
            eval_dataset=downsampled_dataset["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        print("model trained successfully.........")

    def predict(self, text):
        device = "cpu" 

        mask_filler = pipeline(
            "fill-mask", model=self.model.to(device), tokenizer=self.tokenizer
        )

        preds = mask_filler(text)

        return preds, (preds[0]['sequence']), (preds[0]['token_str'])
    

# mlm = MLM(data= "short/100sentences_hin.txt")
# mlm.start()

# text = "किसी अन्य कारण से [MASK] जांच कराने पर इस स्थिति का पता चल सकता है"

# preds, best_pred, best_string = mlm.predict(text)


# print(best_pred)
# print(best_string)