# dataset_util

Helper functions for cleaning Portuguese text in pandas DataFrames before training NLP models.
Written in 2020 for classification experiments on citizen complaint texts.

Everything is in `dataset_util.py`:

- load a CSV and keep the chosen feature columns (`load_data_set`, `clean_data_set`)
- strip HTML tags, accents, punctuation, numbers, e-mails and links
- gensim-based cleaning with an optional full mode that also stems (`clean_text_gensim`)
- sentence tokenization for Portuguese with NLTK

## Requirements

```bash
pip install pandas nltk gensim beautifulsoup4
```

## Example

```python
from dataset_util import clean_text_gensim

clean_text_gensim("<p>O(a) requerente informou que...</p>", full=False)
```
