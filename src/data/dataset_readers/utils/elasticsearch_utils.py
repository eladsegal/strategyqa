import requests
import json
import logging

import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = stopwords.words("english")
STOPWORDS = [stopword + " " for stopword in STOPWORDS]

ELASTICSEARCH_SELECTION_URL = ""  # Needs to be replaced with ID navigation API
ELASTICSEARCH_RETRIEVAL_URL = ""  # Needs to be replaced with retrieval API

logger = logging.getLogger(__name__)


def get_elasticsearch_paragraph(paragraphs_cache, evidence_id):
    if evidence_id not in paragraphs_cache:
        sep_index = evidence_id.rindex("-")
        title = evidence_id[:sep_index]
        para_id = evidence_id[sep_index + 1 :]
        response = requests.get(
            ELASTICSEARCH_SELECTION_URL,
            params={"title": title, "para_id": para_id, "num": 0},
        )
        if response.status_code != 200:
            logger.info(f"Failed to get results from Elasticsearch for: {evidence_id}")
            return None

        paragraphs_cache[evidence_id] = json.loads(response.text)

    if "paras" in paragraphs_cache[evidence_id]:
        paras = paragraphs_cache[evidence_id]["paras"]
        if len(paras) > 0:
            return {
                "evidence_id": evidence_id,
                "title": paras[0]["title"],
                "content": paras[0]["sentence"],
            }
        logger.info(f"Empty results from Elasticsearch for: {evidence_id}")
    else:
        para = paragraphs_cache[evidence_id]
        return {
            "evidence_id": evidence_id,
            "title": para["title"],
            "content": para["content"],
        }


def clean_query(query, remove_stopwords=True):
    if remove_stopwords:
        query_split = query.split()
        new_query_split = []
        for word in query_split:
            if word.lower() + " " not in STOPWORDS:
                new_query_split.append(word)
        query = " ".join(new_query_split)
    return query


def get_elasticsearch_results(queries_cache, query):
    if query not in queries_cache:
        print(f"Missing from elasticsearch cache: {query}")
        response = requests.get(ELASTICSEARCH_RETRIEVAL_URL, params={"concept": query})
        if response.status_code != 200:
            logger.info(f"Failed to get results from Elasticsearch for: {query}")
            return None

        queries_cache[query] = json.loads(response.text)

    paras = queries_cache[query]["paras"]
    if len(paras) > 0:
        return [
            {
                "evidence_id": p["title"] + "-" + str(p["para_id"]),
                "title": p["title"],
                "score": p["score"],
                "section": p["section"],
                "content": p["sentence"],
            }
            for p in paras
        ]
    logger.info(f"Empty results from Elasticsearch for: {query}")
    return None


def concatenate_paragraphs(paragraph_objs):
    special_sep_token = " "
    result = special_sep_token.join([paragraph_obj["content"] for paragraph_obj in paragraph_objs])
    return result
