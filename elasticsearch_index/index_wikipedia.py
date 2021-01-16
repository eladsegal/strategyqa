#!/usr/bin/python3

import argparse
import boto3
import json

from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch.helpers import bulk
from requests_aws4auth import AWS4Auth
from smart_open import open
from tqdm import tqdm

def get_esclient(host, port, region=None):
    if region is not None:
        service= 'es'
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service)
        return Elasticsearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            retries=3,
            timeout=60
        )
    else:
        return Elasticsearch(hosts=[{"host": args.host, "port": args.port}], retries=3, timeout=60)


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(
        description='Add paragraphs from a JSON file to an Elasticsearch index.')
    parser.add_argument('-h', '--host', help='Elastic Search hostname')
    parser.add_argument('-p', '--port', default=9200, help='Port number')
    parser.add_argument('-r', '--region', default=None, help='The region for AWS ES Host. '
                        'If set, we use AWS_PROFILE credentials to connect to the host')
    parser.add_argument('file', help='Path of file to index, e.g. /path/to/my_corpus.json')
    parser.add_argument('index', help='Name of index to create')
    
    args = parser.parse_args()

    # Get Index Name
    index_name = args.index

    # Document Type constant
    TYPE = "paragraph"

    # Get an ElasticSearch client
    es = get_esclient()

    mapping = '''
    {
      "settings": {
        "index": {
          "number_of_shards": 5,
        }
      }
      "mappings": {
        "paragraph": {
          "dynamic": "false",
          "properties": {
            "docId": {
              "type": "keyword"
            },
            "secId": {
               "type": "int"
            }
            "headerid": {
               "type": "keyword"
            }
            "paraId": {
               "type": "int"
            }  
            "title": {
              "analyzer": "snowball",
              "type": "text"
            },
            "section":{
              "analyzer": "snowball",
              "type": "text"
            },
            "header":{
              "analyzer": "snowball",
              "type": "text"
            },
            "text": {
              "analyzer": "snowball",
              "type": "text",
              "fields": {
                "raw": {
                  "type": "keyword"
                }
              }
            },
            "tags": {
              "type": "keyword"
            }
          }
        }
      }
    }'''


    # Function that constructs a json body to add each line of the file to index
    def make_documents(f):
        doc_id = 0
        for l in tqdm(f):
            para_json = json.loads(l)
            doc = {
                '_op_type': 'create',
                '_index': index_name,
                '_type': TYPE,
                '_id': doc_id,
                '_source': {
                    'docId': para_json["docid"],
                    'secId': para_json["secid"],
                    'headerId': para_json["headerid"],
                    'paraId': para_json["para_id"],
                    'title': para_json["title"],
                    'section': para_json.get("section") or "",
                    'subsection': " :: ".join(para_json.get("headers")) or "",
                    'text': para_json["para"]
                }
            }
            doc_id += 1
            yield (doc)


    # Create an index, ignore if it exists already
    try:
        res = es.indices.create(index=index_name, ignore=400, body=mapping)

        # Bulk-insert documents into index
        with open(args.file, "r") as f:
            res = bulk(es, make_documents(f))
            doc_count = res[0]

        print("Index {0} is ready. Added {1} documents.".format(index_name, doc_count))

    except Exception as inst:
        print(inst)
