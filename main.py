import argparse
from copy import deepcopy
import logging

import numpy as np

from src.evaluation import (
    argsort,
    hits_k,
    mrr,
    mutual_nn,
    pw_cosine_similarity,
    scaled_argmin,
)
from src.mapping import iter_norm, procrustes
from src.utils import Dictionary, Vocabulary, list_duplicates

def get_parser() -> argparse.Namespace:
    """
    Initialize CLI.

    Minimum example - UPPERCASE indicate placeholders for file paths.
    python main.py TRAIN_DICO SRC_EMB TRG_EMB --eval_dico EVAL_DICO --dico_delimiter <delimiter_str>
    """

    parser = argparse.ArgumentParser(
            description='(Weakly) supervised bilingual lexicon induction',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # input
    parser.add_argument('train_dico', metavar='PATH', type=str, help='Path to training dictionary')
    parser.add_argument('src_input', metavar='PATH', type=str,
                        help='Path to source embeddings, stored word2vec style')
    parser.add_argument('trg_input', metavar='PATH', type=str,
                        help='Path to target embeddings, stored word2vec style')
    parser.add_argument('--src_output', metavar='PATH', type=str,
                        help='Path to store mapped source embeddings')
    parser.add_argument('--trg_output', metavar='PATH', type=str,
                        help='Path to store mapped target embeddings')
    # input, other 
    parser.add_argument('--vocab_limit', metavar='N', type=int, default=-1,
                        help='Limit vocabularies to top N entries, -1 for all')

    parser.add_argument('--dico_delimiter', metavar='PATH', type=str, default='\t',
                        help='Delimiter in dictionary terms')

    # evaluation
    parser.add_argument('--eval_dico', metavar='PATH', type=str,
                        help='Path to evaluation dictionary')

    # output
    parser.add_argument('--write_dico', metavar='PATH', type=str,
                        help='Write inferred dictionary to path')

    # mapping parameters
    parser.add_argument('-sl', '--self_learning', metavar='N', type=int, default=20,
                        help='Number of self-learning iterations')
    parser.add_argument('-n', '--iter_norm', action='store_true',
                        help='Perform iterative normalization')
    parser.add_argument('-vc', '--vocab_cutoff', metavar='k', nargs='+', type=int,
                        default=20000,
                        help='Restrict self-learning to k most frequent tokens')
    parser.add_argument('--log', metavar="PATH", default="debug", type=str,
                        help='Store log at given path')
    return parser.parse_args()

def setLogger(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(args.log),
            logging.StreamHandler()
        ]
    )

def load_data(args) -> dict:
    """Load all required data into a dictionay."""
    data = {
    'src' : Vocabulary.from_embeddings(args.src_input, top_n_words=args.vocab_limit),
    'trg' : Vocabulary.from_embeddings(args.trg_input, top_n_words=args.vocab_limit),
    'dico' : Dictionary.from_txt(args.train_dico, delimiter=args.dico_delimiter)
    }
    logging.info("============ Data Summary")
    logging.info(f"Source language tokens: {len(data['src'].word2id)}")
    logging.info(f"Target language tokens: {len(data['trg'].word2id)}")
    # lower_case only is a fallback should normal case not be in dictionary
    data['dico'].vocabulary_check(data['src'], data['trg'], lower_case=True)
    if args.eval_dico is not None:
        data['eval_dico'] = Dictionary.from_txt(args.eval_dico,
                                                delimiter=args.dico_delimiter)
        data['eval_dico'].vocabulary_check(data['src'], data['trg'],
                                           lower_case=True)
        logging.info(f"Evaluation pairs: {len(data['eval_dico'].pairs)}")
    return data

# EXPERIMENTAL FEATURE; currently not in use
def validate_eval(data: dict) -> dict:
    """Check that median squared error within dictionary itself has improved."""
    if 'dico_loss'not in data:
        x_w = data['dico'].src_emb
        z_w = data['dico'].trg_emb
        data['dico_loss'] = scaled_argmin(x_w, z_w)
    x_w = data['dico'].src_emb @ data['u']
    z_w = data['dico'].trg_emb @ data['v']
    loss = scaled_argmin(x_w, z_w)
    if loss < data['dico_loss']:
        data['u_argmin'] = data['u'].copy()
        data['v_argmin'] = data['v'].copy()
        data['dico_loss'] = loss
        # print('New argmin', round(loss, 4))
    return data

def iteration(data: dict, args: argparse.Namespace, it: int) -> dict:
    """Perform self-learning iteration. See Vulic et al, 2019, for details."""

    # (1) update embeddings from in-dictionary terms
    data['dico'].update_embeddings(data['src'], data['trg'])

    # (2) solve procrustes for in-dictionary terms
    #     procrustes: SVD of x.T @ z for which u and v are returned
    x = data['dico'].src_emb
    z = data['dico'].trg_emb
    data['u'], data['v'] = procrustes(x, z)

    # (3) map embeddings of to joint space, post-process and select top vocab
    data['vocab_cutoff'] = args.vocab_cutoff[min(len(args.vocab_cutoff)-1, it)]
    x_w = data['src'].emb[:data['vocab_cutoff']] @ data['u']
    z_w = data['trg'].emb[:data['vocab_cutoff']] @ data['v']

    # (4) extend dictionary with mutual NN
    P = pw_cosine_similarity(x_w, z_w)
    src_idx, trg_idx = mutual_nn(src_argmax=P.argmax(1), trg_argmax=P.argmax(0))
    src_mnn = [data['src'].id2word[idx] for idx in src_idx]
    trg_mnn = [data['trg'].id2word[idx] for idx in trg_idx]
    data['dico'].add_tokens(src_mnn, trg_mnn, unique=True)
    data['dico'].update_embeddings(data['src'], data['trg'])
    # data = validate_eval(data)

    logging.info(f'Iteration {it+1} - Dictionary Size: {len(data["dico"].src_tokens)}')
    return data

def candidates_expansion(rankings: np.ndarray, data: dict) -> np.ndarray:
    """Perform evaluation reduction for dictionaries with duplicate terms, e.g. MUSE"""
    # get row indices of duplicate source terms
    duplicates = list_duplicates(data['eval_dico'].src_tokens)
    tokens, pointers = zip(*duplicates)
    # collapse binary indicators of rows:
    # max. since nearest neighbours are solely binary indicators
    for idx in pointers:
        rankings[idx] = rankings[idx].max(0)
    return rankings

def evaluate(data: dict):
    """Evaluate source language against target vocabulary."""
    ref_idx = [data['trg'].word2id[tok] for tok in data['eval_dico'].trg_tokens]
    data['eval_dico'].update_embeddings(data['src'], data['trg'])
    x_w = data['eval_dico'].src_emb @ data['u']
    z_w = data['trg'].emb @ data['v']
    P = pw_cosine_similarity(x_w, z_w)
    rankings = argsort(P, ref_idx)
    # check that duplicates in src language are properly matched
    rankings = candidates_expansion(rankings, data)
    # output
    logging.info("============ Evaluation")
    logging.info(f'MRR: {round(mrr(rankings), 3):.3f}')
    for k in [1, 5, 10]: 
        logging.info(f'HITS@{k}: {round(hits_k(rankings, k), 3):.3f}')

def map_embeddings(data: dict, args: argparse.Namespace):
    """Map and write source and target embeddings."""
    data['src'].emb = data['src'].emb @ data['u']
    data['trg'].emb = data['trg'].emb @ data['v']
    data['src'].write(args.src_output)
    data['trg'].write(args.trg_output)

def write_dico(data: dict, args: argparse.Namespace):
    """Write dictionaries to file. incl-prefix includes pairs from self-learning iterations."""
    x_w = data['src'].emb[:data['vocab_cutoff']] @ data['u']
    z_w = data['trg'].emb[:data['vocab_cutoff']] @ data['v']
    P = pw_cosine_similarity(x_w, z_w)
    src_idx, trg_idx = mutual_nn(src_argmax=P.argmax(1), trg_argmax=P.argmax(0))
    src_mnn = [data['src'].id2word[idx] for idx in src_idx]
    trg_mnn = [data['trg'].id2word[idx] for idx in trg_idx]
    data['out_dico'].add_tokens(src_mnn, trg_mnn, unique=True)
    data['out_dico'].update_embeddings(data['src'], data['trg'])
    logging.info("============ Inferred Dictionaries")
    logging.info(f"Mutual nearest neighbours - excl. SL: {len(data['out_dico'].pairs)}")
    logging.info(f"Mutual nearest neighbours - incl. SL: {len(data['dico'].pairs)}")
    data['out_dico'].to_txt(args.write_dico, args.dico_delimiter)
    data['dico'].to_txt('SL-'+args.write_dico, args.dico_delimiter)

def main():
    args = get_parser()
    setLogger(args)
    # write config
    logging.info(f'============ Config')
    for arg in vars(args):
        logging.info(f'{arg}: {getattr(args, arg)}')

    data = load_data(args)
    data['out_dico'] = deepcopy(data['dico']) # back up dictionary
    if args.iter_norm:
        data['src'].emb = iter_norm(data['src'].emb, axis=[0,1,0,1])
        data['trg'].emb = iter_norm(data['trg'].emb, axis=[0,1,0,1])
        logging.info('Preprocessing: Embeddings iteratively normalized')

    logging.info(f'============ Self-Learning Dictionaries for {args.self_learning} Iterations')
    for it in range(args.self_learning):
        data = iteration(data, args, it)
    if 'eval_dico' in data:
        evaluate(data)
    if args.write_dico:
        write_dico(data, args)
    if args.src_output and args.trg_output:
        map_embeddings(data, args)
        logging.info('============ Embeddings mapped and written to disk')

if __name__ == '__main__':
    main()
