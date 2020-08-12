import numpy as np
import numba as nb
from typing import Tuple

def pw_cosine_similarity(x: np.ndarray, z: np.ndarray, axis: int=-1) -> np.ndarray:
    """
    Compute pairwise cosine similarity for x and z.

    :param x np.ndarray: NxD matrix of stack word embeddings for src lang
    :param z np.ndarray: MxD matrix of stack word embeddings for trg lang
    :param axis int: axis to l2 normalize on
    """
    x /= np.linalg.norm(x, 2, axis=axis, keepdims=True)
    z /= np.linalg.norm(z, 2, axis=axis, keepdims=True)
    return x @ z.T

@nb.njit(parallel=True)
def mutual_nn(src_argmax: np.ndarray, trg_argmax: np.ndarray):
    """
    Infer mutual nearest neighbors.

    Test whether nearest neighbor are mutual by equality check by
    cross-referencing.

    :param src_argmax np.ndarray: nearest target neighbors for source phrases
    :param trg_argmax np.ndarray: nearest source neighbors for target phrases
    """
    N = src_argmax.shape[0]
    M = trg_argmax.shape[0]
    src_argmax = np.stack((np.arange(N), src_argmax), axis=1)
    trg_argmax = np.stack((np.arange(M), trg_argmax), axis=1)
    mutual_neighbours = np.empty(N, dtype=np.bool_)
    for i in nb.prange(N):
        if i == trg_argmax[src_argmax[i, 1], 1]:
            mutual_neighbours[i] = True
        else:
            mutual_neighbours[i] = False
    indices = src_argmax[mutual_neighbours]
    src_idx = indices[:,0]
    trg_idx = indices[:,1]
    return (src_idx, trg_idx)

def scaled_argmin(src_emb, trg_emb) -> float:
    return np.median(np.sqrt(np.linalg.norm(src_emb-trg_emb, ord=2, axis=1)))

def argsort(P: np.ndarray, ref_idx) -> np.ndarray:
    rankings = np.flip(np.argsort(P, axis=-1), axis=-1)
    reference = np.array(ref_idx)[:,None]
    offset = rankings == reference
    return offset

def mrr(rankings: np.ndarray) -> float:
    """
    Compute mean reciprocal rank along axis.

    Presumes correct entries are sorted along rows and columns.

    :param P np.ndarray: pairwise cosine similarity of mapped
                         src and trg embeddings
    :param reverse bool: evaluate on trg language
    """
    N = rankings.shape[-1]
    # reference = np.arange(N)[:,None]
    # offset = rankings == reference
    reciprocal_rank = rankings / np.arange(1, N+1)[None]
    # max along columns to not double count for candidates expansion
    return reciprocal_rank.max(axis=1).mean()

def hits_k(rankings: np.ndarray, k: int) -> float:
    """
    HITS@k for pairwise cosine_similarity matrix.

    Presumes correct entries are sorted along rows and columns.

    :param P np.ndarray: pairwise cosine similarity of mapped
                         src and trg embeddings
    :param k int: test whether reference within top-k
    """
    # N = rankings.shape[0]
    # max along columns to not double count for candidates expansion
    # return (rankings[:,:k] == reference).max(-1).sum() / N
    return rankings[:,:k].max(axis=1).mean()
