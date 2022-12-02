# Gretel Tools

A collection of generalized tools for data science and data engineering.

# Installation

Python 3.6+ required.

1. Clone this repo

2. `pip install -U -I .`

_*or*_

3. `pip install -U -e .` -- if you are editing / making changes

# Header Analysis with FastText Embeddings

**NOTE:** This module will automatically bootstrap itself by
downloading our FastText model from S3

## Get similar Words / Headers

```python
In [33]: h = headers.HeaderAnalyzer(model_file="model.bin.gz")
INFO:gretel_tools.headers:Downloading header model...

In [34]: h.similar_by_word("telephone", topn=5, sort_by = "score")
Out[34]:
[Header(name='telephone*', score=0.9279415011405945, freq=47)
 Header(name='telephone-#', score=0.9162506461143494, freq=109),
 Header(name='telephones', score=0.9141553640365601, freq=32),
 Header(name='phone', score=0.9094647169113159, freq=33660),
 Header(name='telephone-no.', score=0.9073910713195801, freq=361)]

In [35]: h.similar_by_word("telephone", topn=5, sort_by = "freq")
Out[35]:
[Header(name='phone', score=0.9094647169113159, freq=33660),
 Header(name='telephone-no.', score=0.9073910713195801, freq=361),
 Header(name='telephone-#', score=0.9162506461143494, freq=109),
 Header(name='telephone*', score=0.9279415011405945, freq=47),
 Header(name='telephones', score=0.9141553640365601, freq=32)]
```

## Determine Word / Header Similarity

```python
In [36]: h.similarity("sex", "gender")
Out[36]: 0.5893256
```
