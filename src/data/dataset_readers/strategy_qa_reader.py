import json
import logging
import os
from copy import deepcopy
from typing import Optional, List, Dict, Any

from filelock import FileLock

from allennlp.data.fields import MetadataField, LabelField
from overrides import overrides

from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from src.data.dataset_readers.base.base_dataset_reader import BaseDatasetReader
from src.data.fields.dictionary_field import DictionaryField
from src.data.fields.labels_field import LabelsField

from src.data.dataset_readers.utils.elasticsearch_utils import (
    get_elasticsearch_paragraph,
    get_elasticsearch_results,
    concatenate_paragraphs,
    clean_query,
)

logger = logging.getLogger(__name__)

QUERIES_CACHE_PATH = "data/strategyqa/queries_cache.json"


@DatasetReader.register("strategy_qa_reader")
class StrategyQAReader(BaseDatasetReader):
    def __init__(
        self,
        paragraphs_source: str,
        answer_last_decomposition_step: bool = False,
        skip_if_context_missing: bool = True,  # Has an effect only in training
        paragraphs_limit: int = 10,
        generated_decompositions_paths: Optional[List[str]] = None,
        save_elasticsearch_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._paragraphs_source = paragraphs_source
        self._paragraphs_limit = paragraphs_limit
        self._generated_decompositions_paths = generated_decompositions_paths
        self._answer_last_decomposition_step = answer_last_decomposition_step
        self._skip_if_context_missing = skip_if_context_missing

        self._generated_decompositions = None
        if self._generated_decompositions_paths is not None:
            self._generated_decompositions = {}
            for generated_decompositions_path in self._generated_decompositions_paths:
                with open(
                    generated_decompositions_path, mode="r", encoding="utf-8"
                ) as generated_Decompositions_file:
                    for line in generated_Decompositions_file:
                        item = json.loads(line)
                        self._generated_decompositions[item["qid"]] = [
                            {"question": question} for question in item["decomposition"]
                        ]

        self._save_elasticsearch_cache = save_elasticsearch_cache
        with FileLock("cache.lock"):
            self._paragraphs_cache = {}
            if os.path.exists("data/strategyqa/strategyqa_train_paragraphs.json"):
                with open(
                    "data/strategyqa/strategyqa_train_paragraphs.json", "r", encoding="utf8"
                ) as f:
                    self._paragraphs_cache.update(json.load(f))

            if os.path.exists("data/strategyqa/strategyqa_test_paragraphs.json"):
                with open(
                    "data/strategyqa/strategyqa_test_paragraphs.json", "r", encoding="utf8"
                ) as f:
                    self._paragraphs_cache.update(json.load(f))

            if os.path.exists(QUERIES_CACHE_PATH):
                with open(QUERIES_CACHE_PATH, "r", encoding="utf8") as f:
                    self._queries_cache = json.load(f)
            else:
                self._queries_cache = {}

    @overrides
    def _direct_read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading the dataset:")
        logger.info("Reading file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        for json_obj in dataset:
            instance = self._item_to_instance(self.json_to_item(json_obj))
            if instance is not None:
                yield instance

        if self._save_elasticsearch_cache:
            with FileLock("cache.lock"):
                update_required = False
                with open(QUERIES_CACHE_PATH, "r", encoding="utf8") as f:
                    recent_queries_cache = json.load(f)
                    if self._queries_cache != recent_queries_cache:
                        update_required = True
                        self._queries_cache.update(recent_queries_cache)
                if update_required:
                    with open(QUERIES_CACHE_PATH, "w", encoding="utf8") as f:
                        json.dump(self._queries_cache, f)
                        f.flush()

    def json_to_item(self, json_obj):
        question: str = json_obj["question"]
        answer: Optional[bool] = json_obj["answer"] if "answer" in json_obj else None
        facts: Optional[List[str]] = json_obj["facts"] if "facts" in json_obj else None

        if "decomposition" in json_obj:
            decomposition_full = [
                {
                    "question": sub_question,
                    "evidence_per_annotator": [
                        list(
                            dict.fromkeys(
                                [
                                    evidence_id
                                    for evidence_ids in annotator[i]
                                    if isinstance(evidence_ids, list)
                                    for evidence_id in evidence_ids
                                ]
                            )
                        )
                        for annotator in json_obj["evidence"]
                    ],
                    "is_operation_per_annotator": [
                        any(
                            evidence_ids for evidence_ids in annotator[i] if evidence_ids == "operation"
                        )
                        for annotator in json_obj["evidence"]
                    ],
                    "no_evidence_per_annotator": [
                        any(
                            evidence_ids
                            for evidence_ids in annotator[i]
                            if evidence_ids == "no_evidence"
                        )
                        for annotator in json_obj["evidence"]
                    ],
                }
                for i, sub_question in enumerate(json_obj["decomposition"])
            ]
        else:
            decomposition_full = None

        generated_decomposition = None
        if self._generated_decompositions is not None:
            generated_decomposition = self._generated_decompositions[json_obj["qid"]]

        item = deepcopy(json_obj)
        item.update(
            {
                "question": question,
                "answer": answer,
                "decomposition": decomposition_full,
                "generated_decomposition": generated_decomposition,
                "facts": facts,
            }
        )
        return item

    def _item_to_instance(self, item):
        if item is None:
            return None

        question = item["question"]
        answer = item["answer"]
        decomposition = item["decomposition"]
        generated_decomposition = item["generated_decomposition"]
        facts = item["facts"]

        if self._answer_last_decomposition_step:
            question = decomposition[-1]["question"]

        if not self._is_training or answer is not None:
            instance = self.text_to_instance(
                question, answer, decomposition, generated_decomposition, facts
            )
            if instance is not None:
                instance["metadata"].metadata["qid"] = item["qid"]
            return instance
        return None

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        answer: Optional[bool] = None,
        decomposition: Optional[List[Dict[str, Any]]] = None,
        generated_decomposition: Optional[List[Dict[str, Any]]] = None,
        facts: Optional[List[str]] = None,
    ) -> Instance:
        tokenizer_wrapper = self._tokenizer_wrapper
        fields = {}
        pad_token_id = tokenizer_wrapper.tokenizer.pad_token_id

        (context, paragraphs) = self.generate_context_from_paragraphs(
            question=question,
            decomposition=decomposition,
            generated_decomposition=generated_decomposition,
            facts=facts,
        )

        if self._paragraphs_source is not None:
            if context is None:
                if self._is_training and self._skip_if_context_missing:
                    return None
                context = " "
            encoded_input = tokenizer_wrapper.encode(context, question)
        else:
            encoded_input = tokenizer_wrapper.encode(question)

        excluded_keys = ["offset_mapping", "special_tokens_mask"]
        encoded_input_fields = {
            key: LabelsField(value, padding_value=pad_token_id)
            if key == "input_ids"
            else LabelsField(value)
            for key, value in encoded_input.items()
            if key not in excluded_keys
        }
        fields["tokens"] = DictionaryField(
            encoded_input_fields, length=len(encoded_input["input_ids"])
        )

        if answer is not None:
            fields["label"] = LabelField(int(answer), skip_indexing=True)

        # make the metadata

        metadata = {
            "question": question,
            "answer": answer,
            "decomposition": decomposition,
            "generated_decomposition": generated_decomposition,
            "paragraphs": paragraphs["unified"],
            "queries": paragraphs["queries"] if "queries" in paragraphs else [],
        }
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def get_paragraphs(self, **kwargs):
        """
        Returns a list of dictionaries with the following keys:
        "title": str
        "section": str
        "content": str
        "headers": List[str]
        "para_index": int
        """

        paragraphs = {
            None: lambda **kwargs: None,
            "IR-Q": self._ir_q,
            "ORA-P": self._ora_p,
            "IR-ORA-D": self._ir_ora_d,
            "IR-D": self._ir_d,
        }[self._paragraphs_source](**kwargs)

        return paragraphs

    @property
    def generate_context_from_paragraphs(self):
        def f(**kwargs):
            paragraphs = self.get_paragraphs(**kwargs)

            if "unified" not in paragraphs:
                paragraphs.update({"unified": None})
                return None, paragraphs

            if self._paragraphs_limit is not None:
                paragraphs["unified"] = paragraphs["unified"][: self._paragraphs_limit]

            context = concatenate_paragraphs(paragraphs["unified"])
            return context, paragraphs

        return f

    def _ir_q(self, question, **kwargs):
        """
        Get paragraphs from Elasticsearch based on the question and concatenate them
        """
        query = clean_query(question)
        results = get_elasticsearch_results(self._queries_cache, query)
        if results is None:
            return {"queries": [query]}
        return {"unified": results, "queries": [query]}

    def _ora_p(self, decomposition, **kwargs):
        if decomposition is None or len(decomposition) == 0:
            return None

        num_of_matchings = len(decomposition[0]["evidence_per_annotator"])
        if num_of_matchings == 0:
            return None

        all_evidence_ids = [
            [step["evidence_per_annotator"][i] for step in decomposition]
            for i in range(num_of_matchings)
        ]

        evidence_ids = []
        evidence_ids_per_step = [[] for i in range(len(decomposition))]

        # keep it ordered by decomposition steps
        for i in range(len(decomposition)):
            if self._answer_last_decomposition_step:
                # We don't take evidence from the other steps in this setting,
                # as it shouldn't be necessary (in practice, it might be).
                if i < len(decomposition) - 1:
                    continue
            for matching_evidence_ids in all_evidence_ids:
                for evidence_id in matching_evidence_ids[i]:
                    if evidence_id not in evidence_ids_per_step[i]:
                        evidence_ids_per_step[i].append(evidence_id)
                    if evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)

        results = []
        for evidence_id in evidence_ids:
            es_results = get_elasticsearch_paragraph(self._paragraphs_cache, evidence_id)
            if es_results is None:
                return None
            results.append(es_results)
        if len(results) == 0:
            return None

        results_per_step = [[] for i in range(len(decomposition))]
        for i, step_evidence_ids in enumerate(evidence_ids_per_step):
            for evidence_id in step_evidence_ids:
                es_results = get_elasticsearch_paragraph(self._paragraphs_cache, evidence_id)
                if es_results is None:
                    return None
                results_per_step[i].append(es_results)

        return {"unified": results, "per_step": results_per_step}

    def _ir_ora_d(self, decomposition, **kwargs):
        if decomposition is None or len(decomposition) == 0:
            return None

        all_questions = [step["question"] for i, step in enumerate(decomposition)]

        results_per_step = []
        queries = []
        for q in all_questions:
            query = clean_query(q)
            queries.append(query)
            result = get_elasticsearch_results(self._queries_cache, query)
            if result is None:
                return {"queries": queries}
            results_per_step.append(result)

        results = sorted(
            [result for step_results in results_per_step for result in step_results],
            key=lambda item: item["score"],
            reverse=True,
        )

        # Remove duplicates:
        original_results = results
        results = []
        seen_ids = set()
        for result in original_results:
            pid = result["evidence_id"]
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            results.append(result)

        return {"unified": results, "per_step": results_per_step, "queries": queries}

    def _ir_d(self, generated_decomposition, question=None, **kwargs):
        if "decomposition" in kwargs:
            del kwargs["decomposition"]
        return self._ir_ora_d(question=question, decomposition=generated_decomposition, **kwargs)
