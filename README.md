# TensorTonic Solutions

A modular Python library of utility functions for machine learning and numerical computing. Each module provides a focused, dependency-light implementation of a core algorithm — ideal for learning, quick prototyping, or embedding into larger projects.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Distance Metrics](#distance-metrics)
  - [Matrix Operations](#matrix-operations)
  - [Text & NLP Utilities](#text--nlp-utilities)
  - [Probability & Statistics](#probability--statistics)
  - [Transformer Utilities](#transformer-utilities)
- [Module Reference](#module-reference)
- [Getting Help](#getting-help)
- [Contributing](#contributing)

## Features

- **Distance Metrics** — Euclidean, Manhattan, and cosine similarity
- **Matrix Operations** — Dot product, transpose, trace, and diagonal matrix construction
- **Text & NLP Utilities** — Bag-of-words vectorisation, word frequency counting, and sliding-window text chunking
- **Probability & Statistics** — Binomial PMF/CDF, geometric PMF and mean, discrete expected value, and percentile computation
- **Transformer Utilities** — Sinusoidal positional encoding used in attention-based architectures
- Minimal dependencies — only [NumPy](https://numpy.org/) and the Python standard library
- Self-contained modules — import only what you need

## Getting Started

### Prerequisites

- Python 3.7+
- NumPy

```bash
pip install numpy
```

### Installation

Clone the repository:

```bash
git clone https://github.com/anan5093/TensorTonic-Solutions.git
cd TensorTonic-Solutions
```

Because module files are named with hyphens (e.g., `euclidean-distance.py`), use
`importlib` to load them at runtime:

```python
import importlib.util, pathlib

def load_module(name):
    """Load a TensorTonic module by its directory name."""
    path = pathlib.Path(name) / (name + ".py")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

Alternatively, copy any `.py` file into your own project and rename it (replacing
hyphens with underscores) to use a standard `import` statement.

## Usage

### Distance Metrics

```python
ed = load_module("euclidean-distance")
ed.euclidean_distance([0, 0], [3, 4])   # → 5.0

md = load_module("manhattan-distance")
md.manhattan_distance([0, 0], [3, 4])   # → 7.0

cs = load_module("cosine-similarity")
cs.cosine_similarity([1, 0], [1, 1])    # → 0.7071...
```

### Matrix Operations

```python
dp = load_module("dot-product")
dp.dot_product([1, 2, 3], [4, 5, 6])    # → 32.0

mt = load_module("matrix-transpose")
mt.matrix_transpose([[1, 2], [3, 4]])   # → [[1, 3], [2, 4]]

tr = load_module("matrix-trace")
tr.matrix_trace([[1, 2], [3, 4]])    # → 5.0

diag = load_module("make-diagonal")
diag.make_diagonal([1, 2, 3])
# → [[1, 0, 0],
#    [0, 2, 0],
#    [0, 0, 3]]
```

### Text & NLP Utilities

```python
bow = load_module("bag-of-words")
vocab  = ["apple", "banana", "cherry"]
tokens = ["apple", "apple", "banana"]
bow.bag_of_words_vector(tokens, vocab)   # → [2, 1, 0]

wcd = load_module("word-count-dict")
wcd.word_count_dict([["the", "cat"], ["the", "dog"]])
# → {"the": 2, "cat": 1, "dog": 1}

tc = load_module("text-chunking")
tc.text_chunking([1, 2, 3, 4, 5], chunk_size=3, overlap=1)
# → [[1, 2, 3], [3, 4, 5]]
```

### Probability & Statistics

```python
binom = load_module("binomial-pmf-cdf")
pmf, cdf = binom.binomial_pmf_cdf(n=10, p=0.5, k=5)

geo = load_module("geometric-pmf-mean")
pmf_array, mean = geo.geometric_pmf_mean(k=[1, 2, 3, 4, 5], p=0.3)

ev = load_module("expected-value-discrete")
ev.expected_value_discrete(x=[0, 1, 2], p=[0.2, 0.5, 0.3])  # → 1.1

pct = load_module("percentiles")
pct.percentiles([10, 20, 30, 40, 50], q=[25, 50, 75])  # → [20., 30., 40.]
```

### Transformer Utilities

```python
pe = load_module("positional-encoding")
enc = pe.positional_encoding(seq_len=10, d_model=64)
# enc.shape → (10, 64)
# Rows correspond to token positions; columns to model dimensions.
```

## Module Reference

| Directory | Function | Description |
|---|---|---|
| `bag-of-words/` | `bag_of_words_vector(tokens, vocab)` | Frequency vector over a fixed vocabulary |
| `binomial-pmf-cdf/` | `binomial_pmf_cdf(n, p, k)` | Binomial PMF and CDF at `k` successes |
| `cosine-similarity/` | `cosine_similarity(a, b)` | Cosine similarity between two vectors |
| `dot-product/` | `dot_product(x, y)` | Scalar dot product of two 1-D arrays |
| `euclidean-distance/` | `euclidean_distance(x, y)` | L2 distance between two vectors |
| `expected-value-discrete/` | `expected_value_discrete(x, p)` | Expected value of a discrete distribution |
| `geometric-pmf-mean/` | `geometric_pmf_mean(k, p)` | Geometric PMF array and distribution mean |
| `make-diagonal/` | `make_diagonal(v)` | Diagonal matrix from a 1-D vector |
| `manhattan-distance/` | `manhattan_distance(x, y)` | L1 distance between two vectors |
| `matrix-trace/` | `matrix_trace(A)` | Sum of the main-diagonal elements |
| `matrix-transpose/` | `matrix_transpose(A)` | Transpose of a 2-D matrix |
| `percentiles/` | `percentiles(x, q)` | Percentile values via linear interpolation |
| `positional-encoding/` | `positional_encoding(seq_len, d_model, base)` | Sinusoidal positional encoding matrix |
| `text-chunking/` | `text_chunking(tokens, chunk_size, overlap)` | Sliding-window token chunking |
| `word-count-dict/` | `word_count_dict(sentences)` | Word frequency dictionary from sentence lists |

## Getting Help

- **Bug reports & feature requests** — open an issue on the [GitHub Issues](https://github.com/anan5093/TensorTonic-Solutions/issues) page.
- **Questions** — use the [GitHub Discussions](https://github.com/anan5093/TensorTonic-Solutions/discussions) tab for general questions or ideas.

## Contributing

Contributions are welcome! To add a new module or improve an existing one:

1. Fork the repository and create a feature branch.
2. Add your module in a new directory following the existing naming convention (`kebab-case/`).
3. Ensure the implementation uses only NumPy (or the Python standard library) and handles edge cases gracefully.
4. Open a pull request with a clear description of what the module does and example usage.

Please keep each module focused on a single responsibility and match the code style of existing files.
