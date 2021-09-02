import argparse
import os
from typing import List, Union

import numpy as np


def iter_norm(arr: np.ndarray, k: int=5, norm: int=2,
              axis: Union[List[int], int]=[0,1,0,1]):
    """
    Iteratively mean and 'norm' normalize arr for k iterations (each) along given axes.
    The below implementation differs from the original github;
    however it is more true to the original algorithm as presented in the paper.
    The relevant properties:
        (1) Each word embedding is unit length
        (2) The average word embedding is zero (emb.mean(0) = 0)
        (3) Each word embedding has mean zero (emb.mean(1) = 0)

    More emphasis on condition 2 or 3 can be put by setting 0 or 1,
    respectively, as last axis to normalize over.

    :param arr np.ndarray: arr to normalize
    :param k int: number of iterations
    :param norm int: vector norm to normalize entries for
    :param axis Union[List[int], int]: series of axes to normalize on
    """
    if isinstance(axis, list):
        for ax in axis:
            arr = iter_norm(arr, k=k, norm=norm, axis=ax)
    elif isinstance(axis, int):
        for _ in range(k):
            arr /= np.linalg.norm(arr, axis=axis, ord=norm, keepdims=True)
            arr -= arr.mean(axis=axis, keepdims=True)
    return arr

def procrustes(x: np.ndarray, y: np.ndarray):
    """
    Compute transformation matrices u and v to project x and y into shared
    space, respectively.

    See section 2.2 C2 Self-Learning of [Vulić et al] for more details.
    Vulić et al, EMNLP, 2019,
    Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?
    https://www.aclweb.org/anthology/D19-1449/

    :param x np.ndarray: stacked in-dico word embeddings of src lang
    :param y np.ndarray: stacked in-dico word embeddings of trg lang
    """
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    return (u @ vt).T
