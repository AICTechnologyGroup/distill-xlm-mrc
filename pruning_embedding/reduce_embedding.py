import torch
from transformers import AutoTokenizer, AutoModel
import argparse

def reduce_embedding(old_tokenizer, small_vocab, model):

    kept_ids = []
    for w in small_vocab:
        kept_ids.append(old_tokenizer.convert_tokens_to_ids(w))

    old_embeddings = model.get_input_embeddings()
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_size = len(kept_ids)
    new_embeddings = torch.nn.Embedding(new_size, old_embedding_dim)
    # new_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=new_size, bias=False)
    for new_id, old_id in enumerate(kept_ids):
        new_embeddings.weight.data[new_id, :] = old_embeddings.weight.data[old_id, :]

    model.set_input_embeddings(new_embeddings)

    # Update base model and current model config
    model.config.vocab_size = new_size
    model.vocab_size = new_size

    # Tie weights
    model.tie_weights()
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path',required=True,type=str)
    parser.add_argument('--small_vocab_path',required=True,type=str)
    parser.add_argument('--save_model_path',required=True,type=str)

    args,_ = parser.parse_known_args()
    old_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    

    with open(args.small_vocab_path,'r') as f:
        small_vocab = f.read().splitlines()
    
    new_model = reduce_embedding(old_tokenizer, small_vocab, model)
    new_model.save_pretrained(args.save_model_path)