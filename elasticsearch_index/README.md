 # Elasticsearch Index
 This script uses the Python Elasticsearch API to index a para corpus in an Elasticsearch cluster.  
 The corpus is expected to be a JSONL file with each line containing a JSON object provided in [enwiki-20200511-cirrussearch-parasv2.jsonl.gz](https://storage.googleapis.com/ai2i/strategyqa/data/corpus-enwiki-20200511-cirrussearch-parasv2.jsonl.gz).  
 See make_documents for the mapping of this JSON into the index fields.  
 If an index with the requested name does not exists, creates it.  
 If not, simply adds documents to existing index.

### Requirements
```
cd elasticsearch_index
pip install -r requirements.txt
```

### Indexing
 Sample command:
 ```
 python index_wikipedia.py -h localhost enwiki-20200511-cirrussearch-parasv2.jsonl.gz strategyqa
 ```

Sample command using AWS ES service:
```
AWS_PROFILE=<profile name> python index_wikipedia.py \
-h <URL of AWS ES service> -r <aws region> \
enwiki-20200511-cirrussearch-parasv2.json.gz strategyqa
```