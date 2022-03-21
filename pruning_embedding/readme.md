Reduced size  pretrained language model as follows [Load What You Need](https://arxiv.org/abs/2010.05609)


```shell
python reduce_embedding.py --model_name_or_path xlm-roberta-base \
                           --small_vocab_path small_vocab.txt \
                           --save_model_path small_model \
```