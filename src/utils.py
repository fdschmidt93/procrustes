from collections import defaultdict
from dataclasses import dataclass, field
import os
import pickle
from typing import Dict, Optional

import numpy as np

def list_duplicates(tokens):
    tally = defaultdict(list)
    for i,item in enumerate(tokens):
        tally[item].append(i)
    return [(key,locs) for key,locs in tally.items() if len(locs)>1]

@dataclass
class Vocabulary:
    """
    Ancillary word2id, id2word, and embeddings container class.

    :attr len: int, number of tokens in Dictionary
    :attr word2id: dict, keys: tokens, values: id
    :attr id2word: dict, k-v pair inverse of word2id
    :attr embeddings: list[list[np.ndarray]], tokens embedded
                      -- N: number of words
                      -- d: word vector dimensionality
    """
    word2id: dict = field(default_factory=dict)
    id2word: dict = field(default_factory=dict)
    emb: np.ndarray = field(init=False)

    def __len__(self):
        return len(self.word2id)

    def __getitem__(self, index):
        return self.id2word[index]

    def add(self, word):
        """
        Add a new word to word2id & id2word.

        :param word:
            (a) str, word to be added, e.g. "hello"
            (b) list[str], words to be added, e.g. ["first", "second"]
        """
        if isinstance(word, list):
            for token in word:
                self.add(token)
        else:
            assert isinstance(word, str), "Passed argument not a string"
            if word not in self.word2id:
                len_ = len(self)
                self.word2id[word] = len_
                self.id2word[len_] = word

    def add_embeddings(self, embeddings: np.ndarray, dtype=np.float32):
        assert isinstance(embeddings, np.ndarray), 'Embeddings are no numpy arrays!'
        assert len(embeddings) == len(self.word2id), 'Length mismatch!'
        self.emb = np.asarray(embeddings, dtype=dtype)

    @classmethod
    def from_embeddings(cls,
                        path: str,
                        pass_header: bool = True,
                        top_n_words: int = 200_000,
                        normalize: bool = False,
                        dtype=np.float32):
        """
        Instantiate Dictionary from pretrained word embeddings.

        :param path: str, path to pretrained word embeddings
        :param top_n_words: int, restrict Dictionary top_n_words frequent words
        :return: Dictionary, populated word2id and id2word from document tokens
        """
        assert os.path.exists(path), f'{path} not found!'
        cls_ = cls()
        with open(path, 'r') as f:
            embeddings = []
            if pass_header:
                next(f)
            for idx, line in enumerate(f):
                if len(cls_) == top_n_words:
                    break
                token, vector = line.rstrip().split(' ', maxsplit=1)
                if token not in cls_.word2id:
                    embeddings.append(np.fromstring(vector.strip(), sep=' '))
                    cls_.add(token)
        cls_.emb = np.asarray(np.stack(embeddings), dtype=dtype)
        if normalize:
            norm = np.linalg.norm(cls_.emb, ord=2, axis=-1, keepdims=True)
            cls_.emb /= norm
        assert len(cls_.emb) == len(cls_.word2id), 'Reading error!'
        return cls_

    def phrase_embeddings(self, path_to_phrases: str):
        phrases = []
        with open(path_to_phrases, 'r') as file:
            for line in file:
                tokens = line.strip()
                if tokens:
                    if ' ' in tokens:
                        tokens = tokens.split(' ')
                    phrases.append(tokens)

        # reorganize variables
        word2id = {}
        id2word = {}
        embs = []
        for phrase in phrases:
            # unigram
            if isinstance(phrase, str):
                pointers = self.word2id[phrase]
                emb = self.emb[pointers]
            elif isinstance(phrase, list):
                pointers = [self.word2id[tok] for tok in phrase]
                phrase = '&#32;'.join(phrase)
                emb = self.emb[pointers].mean(0)
            # filter final edge cases starting with space, etc.
            if not phrase in word2id:
                id_ = len(word2id)
                word2id[phrase] = id_
                id2word[id_] = phrase
                embs.append(emb)
        self.word2id = word2id
        self.id2word = id2word
        self.emb = np.stack(embs)

    @classmethod
    def from_dictionary(cls,
                        dict_: Dict[str, int],
                        embeddings: Optional[np.ndarray] = None):
        """Instantiate Vocabulary instance from word2id dictionary."""
        assert isinstance(dict_, dict), 'Please pass a dictionary!'
        cls_ = cls()
        cls_.word2id = dict_
        cls_.id2word = {v: k for k, v in dict_.items()}
        if embeddings is not None:
            assert len(cls_) == len(embeddings), 'Shapes do not align!'
            assert isinstance(embeddings, np.ndarray), 'Not an np.ndarray'
            cls_.emb = embeddings
        return cls_

    @classmethod
    def from_pretrained(cls,
                        word2id_path: str,
                        embeddings_path: str):
        cls_ = cls()
        with open(word2id_path, 'rb') as file:
            word2id = pickle.load(file)
            cls_.word2id = word2id

        cls_.id2word = {v: k for k, v in cls_.word2id.items()}
        cls_.emb = np.load(embeddings_path)
        return cls_

    def write(self, path, header=True):
        with open(path, 'w') as vec:
            if header:
                header_string = f'{len(self.word2id)} {self.emb.shape[-1]}'
                vec.write(header_string+'\n')
            for i, (word, emb) in enumerate(zip(self.word2id, self.emb)):
                if (i + 1) % 100_000 == 0:
                    print(f'{i+1} of {len(self.word2id)} phrases written to file!')
                out = word + ' ' + ' '.join(map(str, emb))
                vec.write(out+'\n')

@dataclass
class Dictionary:
    """
    Ancillary class to store training and evaluation dictionaries.

    :attr src_tokens: list, ordered token that translates to corresponding trg
    :attr trg_tokens: list, ordered token that translates to corresponding src
    :attr pairs: list, list of tuples of corresponding src and trg translations
    :attr src_emb: np.ndarray, word embeddings for tokens in src_tokens
    :attr trg_emb: np.ndarray, word embeddings for tokens in trg_tokens
    """
    src_tokens : list = field(default_factory=list)
    trg_tokens : list = field(default_factory=list)
    pairs : list = field(default_factory=list)
    src_emb : np.ndarray = field(init=False)
    trg_emb : np.ndarray = field(init=False)
    src_duplicates : list = field(default_factory=list)

    @classmethod
    def from_txt(cls, path: str, delimiter: str=' '):
        cls_ = cls()
        with open(path) as dict_:
            for line in dict_:
                src_token, trg_token = line.rstrip().split(delimiter)
                cls_.src_tokens.append(src_token)
                cls_.trg_tokens.append(trg_token)
        cls_.pairs = list(zip(cls_.src_tokens, cls_.trg_tokens))
        return cls_

    @classmethod
    def from_tokens(cls, src, trg, unique=False):
        cls_ = cls()
        cls_.add_tokens(src, trg, unique=unique)
        return cls_

    def update_embeddings(self, src_: Vocabulary, trg_: Vocabulary):
        def update(tokens, vocab: Vocabulary):
            pointers = [vocab.word2id[token] for token in tokens]
            return vocab.emb[pointers].copy()
        self.src_emb = update(self.src_tokens, src_)
        self.trg_emb = update(self.trg_tokens, trg_)


    def add_tokens(self, src, trg, unique=False):
        if isinstance(src, list):
            assert isinstance(trg, list), 'trg is no list!'
            assert len(src) == len(trg), 'Mismatch in words to add!'
            for src_token, trg_token in zip(src, trg):
                self.add_tokens(src_token, trg_token, unique=unique)
        elif isinstance(src, str):
            pair_exists = False
            assert isinstance(trg, str), 'Mismatch in words to add!'
            if unique:
                pair = (src, trg)
                pair_exists = pair in self.pairs
            else:
                pair_exists = False
            if not pair_exists:
                self.src_tokens.append(src)
                self.trg_tokens.append(trg)
                self.pairs.append((src, trg))
        self.check()

    def check(self):
        """Check if lengths of dictionaries align."""
        assert len(self.src_tokens) == len(self.trg_tokens) == len(self.pairs)

    def vocabulary_check(self,
                         src: Vocabulary, trg: Vocabulary,
                         lower_case: bool=True):
        """
        Check alignment of dictionary terms with embeddings vocabulary.

        If lower_case is True, the function will fall back to lower casing in
        case original casing is not found in vocabulary.

        :param src Vocabulary: Vocabulary of source lang embeddings
        :param trg Vocabulary: Vocabulary of target lang embeddings
        :param lower_case bool: whether to fall-back to lower casing
        """
        def consistency_check(tokens: list, vocab: Vocabulary, lower_case: bool):
            out = []
            for token in tokens:
                if token in vocab.word2id:
                    out.append(token)
                elif (token.lower() in vocab.word2id) and lower_case:
                    out.append(token.lower())
                else:
                    raise KeyError(f'{token} not found in respective Vocabulary!')
            return out
        self.src_tokens = consistency_check(self.src_tokens, src, lower_case)
        self.trg_tokens = consistency_check(self.trg_tokens, trg, lower_case)
        self.pairs = list(zip(self.src_tokens, self.trg_tokens))

    def to_txt(self, path:str, delimiter:str=' '):
        assert isinstance(delimiter, str), 'Pass valid delimiter!'
        with open(path, 'w') as file:
            for src_, trg_ in self.pairs:
                file.write(f'{src_}{delimiter}{trg_}\n')
