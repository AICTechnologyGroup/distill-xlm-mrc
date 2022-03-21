from packaging.version import parse
from transformers import TrainingArguments
from model import MRCQuestionAnswering
from transformers import Trainer
from utils import data_loader
import numpy as np
from datasets import load_metric
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path')
    parser.add_argument('--output_dir')
    args,_ = parser.parse_known_args()

    model = MRCQuestionAnswering.from_pretrained(args.pretrained_path)
    print(model)
    print(model.config)

    train_dataset, valid_dataset = data_loader.get_dataloader(
        train_path='./data-bin/processed/train.dataset',
        valid_path='./data-bin/processed/valid.dataset'
    )

    training_args = TrainingArguments(args.output_dir,
                                      do_train=True,
                                      do_eval=True,
                                      num_train_epochs=10,
                                      learning_rate=1e-4,
                                      warmup_ratio=0.05,
                                      weight_decay=0.01,
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=32,
                                      gradient_accumulation_steps=1,
                                      logging_dir='./log',
                                      logging_steps=350,
                                      label_names=['start_positions',
                                                   'end_positions',
                                                   'span_answer_ids',
                                                   'input_ids',
                                                   'words_lengths'],
                                      group_by_length=True,
                                      save_strategy="epoch",
#                                       save_strategy = "steps",
                                      metric_for_best_model='f1',
                                      load_best_model_at_end=True,
#                                       save_steps=350,
#                                       save_total_limit=2,
#                                       eval_steps=1,
#                                       evaluation_strategy="steps",
                                      evaluation_strategy="epoch",
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_loader.data_collator,
        compute_metrics=data_loader.compute_metrics
    )

    trainer.train()
