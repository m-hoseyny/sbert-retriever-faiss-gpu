# SBERT Retriever - FAISS GPU

This repository allows you to use the SBERT retriever on a GPU with FAISS.

Test: 
*On a GTX A1000 with MSMarco v1 passages, it achieves 3.3ms per query.*


## Install requirements

Please note that older FAISS versions are not compatible with Python > 3.8. Therefore, please install Python 3.8.x on your system. (Using a Conda environment is recommended.)

Clone the project:
```
git clone https://github.com/m-hoseyny/sbert-retriever-faiss-gpu
```

```
pip install -r requirements.txt
```

## Index the Documents
Due to resource limitations, the code uses a chunking mechanism. For each chunk, the model encodes documents and creates a FAISS shard for it. This helps to save RAM usage.
```
python indexer.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --documents collection.tsv \
    --output content/ \
    --chunks-size 500000 \
    --device cuda:0 
```

## Retrieve
```
python retriever.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --lookup-file content/lookup_file.pk \
    --topics train.queries.tsv \
    --chunks-size 500000 \
    --device cuda:0 \ 
    --hits 10 \
    --output run.sbert.gpu.train.queries.tsv
```