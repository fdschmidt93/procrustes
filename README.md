# procrustes

procrustes is an open source framework to induce cross-lingual word embeddings with (weakly) supervised by dictionaries. The framework is inspired by [vecmap](https://github.com/artetxem/vecmap), though improves on specific aspects of the implementation, namely speed-ups via vectorization, self-learning restricted to mutual nearest neighbors, and optional preprocessing via iterative normalization. The framework does not support fully unsupervised mappings, as lack thereof has been shown to collapse BLI performance for distant language pairs [[1](https://arxiv.org/pdf/1909.01638.pdf)]. Instead, we opt for expanding seeding supervision, for instance sourced by [plexy](https://github.com/fdschmidt93/plexy) or [MUSE](https://github.com/facebookresearch/MUSE), via self-learning.

## Features

* **Transparent:** Clean implementation of self-learning-based induction of cross-lingual embedding spaces
* **Quality:** Self-learning via extending seeding dictionary only with mutual nearest neighbors
* **Fast:** All computation heavy aspects of pipeline vectorized and accelerated
* **Feature-rich:** Evaluate, expand seeding dictionary, and map embeddings in single execution

## Seeding Dictionaries

Since the framework does not include a fully unsupervised mode, seeding supervision is required to induce a mapping. To that end, two sources are available:

1. [MUSE](https://github.com/facebookresearch/MUSE) provides 110 uncased ground-truth bilingual dictionaries
2. [plexy](https://github.com/fdschmidt93/plexy) constructs a bilingual lexicon by querying [PanLex](https://panlex.org/), a panlinugal lexicon comprising 2,500 dictionaries of 5,700 languages

## Walkthrough
See [Vulic et al., EMNLP 2019,](https://arxiv.org/pdf/1909.01638.pdf) for a detailed walkthrough of the self-learning-based pipeline:

1. Load embeddings and seeding dictionary for source and target language
2. Optional: apply [iterative normalization](https://arxiv.org/pdf/1906.01622.pdf) to boost BLI performance
3. Self-learning - for each iteration do: 
    1. Induce orthogonal mapping with current dictionary 
    2. Expand current dictionary with unique mutual nearest neighbors
5. Evaluation (MRR, HITS@{1,5,10}), dictionary extraction, and mapping full embeddings


## Requirements

* Python 3
* NumPy
* Numba

## Command-line Interface

```
(Weakly) supervised bilingual lexicon induction

positional arguments:
  PATH                     Path to training dictionary
  PATH                     Path to source embeddings, stored word2vec style
  PATH                     Path to target embeddings, stored word2vec style

optional arguments:
  --src_output PATH        Path to store mapped source embeddings (default: None)
  --trg_output PATH        Path to store mapped target embeddings (default: None)
  --vocab_limit N          Limit vocabularies to top N entries, -1 for all (default: -1)
  --dico_delimiter PATH    Delimiter in dictionary terms (default: tab-delimited)
  --eval_dico PATH         Path to evaluation dictionary (default: None)
  --write_dico PATH        Write inferred dictionary to path (default: None)
  --self_learning N        Number of self-learning iterations (default: 20)
  --iter_norm              Perform iterative normalization (default: False)
  --vocab_cutoff k [k ...] Restrict self-learning to k most frequent tokens (default: 20000)
  --log PATH               Store log at given path (default: debug)
```
### Usage
* See `evaluation.sh` for an example configuration to run evaluation and `ko-eo.example.txt` for an illustrative output log.

### CLI Comments 
* **write_dictionary:** two dictionaries are written to disk -- 
    1. Dictionary expanded by self-learning, prefixed with "SL"
    2. Expand seeding dictionary anew by mutual neearest neighbors with learned mapping
* **vocab_cutoff:** can be set as a list with cutoff ramp-up, e.g. 500 2500 5000, where remaining iterations use last value

## References
 
### Resources
* [PanLex](https://panlex.org/), the world's largest lexical database establishing a panlingual dictionary covering 5,700 languages

### Papers
* Ivan Vulić, Goran Glavaš, Roi Reichart, Anna Korhonen, [*Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?, EMNLP 2019*](https://arxiv.org/abs/1909.01638)
* Mozhi Zhang, Keyulu Xu, Ken-ichi Kawarabayashi, Stefanie Jegelka, Jordan Boyd-Graber [*Are Girls Neko or Shōjo? Cross-Lingual Alignment of Non-Isomorphic Embeddings with Iterative Normalization*](https://arxiv.org/abs/1909.01638)

## Contact

**Author:** Fabian David Schmidt\
**Affiliation:** University of Mannheim\
**E-Mail:** fabian.david.schmidt@hotmail.de
