# distill-xlm-mrc
A distill model for machine reading comprehension

## Model Description

- Language model: [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base), [MiniLMv2](https://huggingface.co/nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large)
- Fine-tune: [MRCQuestionAnswering](https://github.com/nguyenvulebinh/extractive-qa-mrc)
- Language: Vietnamese, Englsih
- Downstream-task: Extractive QA


## Training singgle model
In data-bin/raw folder already exist some sample data files for the training process. Do following steps:

- Create environment by using file requirements.txt

- Clean data

```shell
python squad_to_mrc.py
python train_valid_split.py
```
- Train model

```shell
python singgle_training.py --output_dir <output directory>\
                           --pretrained_path <pretrained model>
```

- Train KD model<br>
You shold train teacher and student model first

```shell
python singgle_training.py --output_dir <output teacher directory>\
                           --pretrained_path <pretrained teacher model>
```

```shell
python singgle_training.py --output_dir <output student directory>\
                           --pretrained_path <pretrained student model>
```

```shell
python kd_training.py --output_dir <output directory>\
                           --pretrained_student_path <pretrained student model>\
                           --pretrained_teacher_path <pretrained student model>\
                           --distill_hardness 0.5\
                           --distill_temperature 2\
```

Reduced size  pretrained language model as follows [Load What You Need](https://arxiv.org/abs/2010.05609)


```shell
python pruning_embedding/reduce_embedding.py --model_name_or_path <pretrained student model> \
                           --small_vocab_path small_vocab.txt \
                           --save_model_path small_model \
```
### Using pre-trained model

```python
from transformers import pipeline
model_checkpoint = "aicryptogroup/distill-xlm-mrc"
nlp = pipeline('question-answering', model=model_checkpoint,
                   tokenizer=model_checkpoint)
QA_input = {
  'question': "what is the capital of Vietnam",
  'context': "Keeping an ageless charm through centuries, Hanoi - the capital of Vietnam is famous not only for the Old Quarter with narrow and crowded streets but also for the nostalgic feeling that it brings. While Saigon is a young and modern city, the ancient Hanoi is still a true beholder of history."
}
res = nlp(QA_input)
print('pipeline: {}'.format(res))
```
### Acknowledgement
Our code is based on  implementation of [nguyenvulebinh](https://github.com/nguyenvulebinh/extractive-qa-mrc)