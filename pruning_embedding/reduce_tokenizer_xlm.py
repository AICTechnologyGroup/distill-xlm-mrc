from transformers import AutoTokenizer, XLMRobertaTokenizer
from sentencepiece import sentencepiece_model_pb2 as spmp
import os
import argparse



def reduce_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    with open(args.small_vocab_path) as f:
        vocab = f.read().splitlines()

    kept_ids = []
    for w in vocab:
        kept_ids.append(tokenizer.convert_tokens_to_ids(w))
    m = spmp.ModelProto()
    m.ParseFromString(open(os.path.join(args.model_name_or_path,"sentencepiece.bpe.model"), 'rb').read())
    new_pieces = [m.pieces[idx-1] for idx in kept_ids[4:]]
    for i, p in enumerate(new_pieces):
        m.pieces[i+3].piece = p.piece
        m.pieces[i+3].score = p.score
        m.pieces[i+3].type = p.type
    n = len(new_pieces)

    i = len(m.pieces)
    while i> n+3:
        x = m.pieces.pop()
        i-=1

    with open(os.path.join(args.save_model_path,"sentencepiece.bpe.model"), 'wb') as f:
        f.write(m.SerializeToString())

    new_tokenizer = XLMRobertaTokenizer(os.path.join(args.save_model_path,"sentencepiece.bpe.model"))
    new_tokenizer.save_pretrained(args.save_model_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path',required=True,type=str)
    parser.add_argument('--small_vocab_path',required=True,type=str)
    parser.add_argument('--save_model_path',required=True,type=str)

    args,_ = parser.parse_known_args()
    reduce_tokenizer(args)








