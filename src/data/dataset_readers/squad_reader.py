import json
import logging
from typing import Any, Dict, List, Optional, Iterable

from allennlp.data.fields import MetadataField, SpanField
from overrides import overrides
import functools

from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from src.data.dataset_readers.base.base_dataset_reader import BaseDatasetReader
from src.data.tokenizers.offset_mapping_utils import (
    get_sequence_boundaries,
    get_token_answer_span,
    group_tokens_by_whole_words,
    find_valid_spans,
)
from src.data.fields.dictionary_field import DictionaryField
from src.data.fields.labels_field import LabelsField

from src.data.dataset_readers.utils.standardization_utils import standardize_text_simple

from src.data.types import Span

logger = logging.getLogger(__name__)


@DatasetReader.register("general_squad")
class SquadV1Reader(BaseDatasetReader):
    def __init__(self, length_limit: int = 512, stride: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)

        self._length_limit = length_limit
        self._stride = stride

    @overrides
    def _reader_specific_init(self):
        self.additional_special_tokens.add("@@YES_NO_SEP@@")

    @overrides
    def _direct_read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
        logger.info("Reading the dataset")
        yielded_question_count = 0
        questions_with_more_than_one_instance = 0
        for article in dataset:
            for paragraph_json in article["paragraphs"]:
                context, context_offset = standardize_text_simple(
                    paragraph_json["context"], output_offset=True
                )
                for question_answer in paragraph_json["qas"]:
                    answers = [
                        standardize_text_simple(answer_json["text"])
                        for answer_json in question_answer["answers"]
                    ]

                    # Just like huggingface, we only use the first answer for training.
                    is_impossible = question_answer.get("is_impossible", None)
                    if len(answers) > 0:
                        first_answer_start_offset = int(
                            question_answer["answers"][0]["answer_start"] + context_offset
                        )
                    else:
                        if is_impossible:
                            answers.append("")
                        first_answer_start_offset = None

                    instances = self.make_instances(
                        standardize_text_simple(question_answer["question"]),
                        context,
                        answers,
                        first_answer_start_offset,
                        question_answer.get("is_impossible", None),
                        question_answer.get("id", None),
                        "is_boolq" in question_answer,
                    )
                    instances_yielded = 0
                    for instance in instances:
                        yield instance
                        instances_yielded += 1
                    if instances_yielded > 1:
                        questions_with_more_than_one_instance += 1
                    yielded_question_count += 1

        if questions_with_more_than_one_instance > 0:
            logger.info(
                "%d (%.2f%%) questions have more than one instance",
                questions_with_more_than_one_instance,
                100 * questions_with_more_than_one_instance / yielded_question_count,
            )

    def make_instances(
        self,
        question: str,
        context: str,
        answers: Optional[List[str]] = None,
        first_answer_start_offset: Optional[int] = None,
        is_impossible: Optional[bool] = None,
        qid: Optional[str] = None,
        is_boolq: bool = False,
    ) -> Iterable[Instance]:

        modified_question = "yes no@@YES_NO_SEP@@" + question + ("?" if is_boolq else "")

        encoded_input = self._tokenizer_wrapper.encode(
            modified_question,
            context,
            truncation="only_second",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_overflowing_tokens=True,
            max_length=self._length_limit,
            stride=self._stride,
        )

        if is_boolq:
            first_answer_start_offset = modified_question.index(answers[0])

        num_of_windows = len(encoded_input["overflow_to_sample_mapping"])
        for i in range(num_of_windows):
            instance = self.text_to_instance(
                question,
                modified_question,
                context,
                {
                    key: value[i]
                    for key, value in encoded_input.items()
                    if len(value) == num_of_windows
                },
                answers,
                first_answer_start_offset,
                is_impossible,
                qid,
                i,
                is_boolq,
            )
            if instance is not None:
                yield instance
            break  # Don't use windows, just the first

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        modified_question: str,
        context: str,
        encoded_input: Dict[str, Any],
        answers: List[str] = None,
        first_answer_start_offset: Optional[int] = None,
        is_impossible: Optional[bool] = None,
        qid: str = None,
        window_index: int = None,
        is_boolq: bool = False,
    ) -> Instance:
        tokenizer_wrapper = self._tokenizer_wrapper
        offset_mapping = encoded_input["offset_mapping"]
        special_tokens_mask = encoded_input["special_tokens_mask"]

        token_answer_span = None
        if first_answer_start_offset is not None and answers:
            answer = answers[0]
            relevant_sequence_index = 0 if is_boolq else 1
            tokens_groups = group_tokens_by_whole_words(
                [modified_question, context], offset_mapping, special_tokens_mask
            )
            valid_spans = find_valid_spans(
                modified_question if is_boolq else context,
                answer,
                offset_mapping,
                special_tokens_mask,
                functools.partial(
                    get_token_answer_span,
                    sequence_index=relevant_sequence_index,
                ),
                tokens_groups,
                first_answer_start_offset,
            )
            token_answer_span = valid_spans[0] if len(valid_spans) > 0 else None

        if self._is_training and (
            token_answer_span is None and (is_impossible is None or is_impossible is False)
        ):
            return None

        seq_boundaries = get_sequence_boundaries(special_tokens_mask)
        (first_context_token_index, last_context_token_index,) = (
            seq_boundaries[1] if len(seq_boundaries) > 1 else (-1, -1)
        )

        fields = {}
        pad_token_id = tokenizer_wrapper.tokenizer.pad_token_id

        excluded_keys = [
            "offset_mapping",
            "special_tokens_mask",
            "overflow_to_sample_mapping",
        ]

        encoded_input_fields = {
            key: LabelsField(value, padding_value=pad_token_id)
            if key == "input_ids"
            else LabelsField(value)
            for key, value in encoded_input.items()
            if key not in excluded_keys
        }
        fields["question_with_context"] = DictionaryField(
            encoded_input_fields, length=len(encoded_input["input_ids"])
        )

        # make the answer span
        seq_field = encoded_input_fields["input_ids"]
        if token_answer_span is not None:
            assert all(i >= 0 for i in token_answer_span)
            assert token_answer_span.start <= token_answer_span.end

            fields["answer_span"] = SpanField(
                token_answer_span.start,
                token_answer_span.end,
                seq_field,
            )
        else:
            # We have to put in something even when we don't have an answer, so that this instance can be batched
            # together with other instances that have answers.
            if is_impossible is True:
                fields["answer_span"] = SpanField(0, 0, seq_field)
            else:
                fields["answer_span"] = SpanField(-1, -1, seq_field)

        # make the context span, i.e., the span of text from which possible answers should be drawn
        fields["context_span"] = SpanField(
            first_context_token_index,
            last_context_token_index,
            seq_field,
        )
        fields["yes_no_span"] = SpanField(
            seq_boundaries[0][0],
            seq_boundaries[0][0] + 1,
            seq_field,
        )

        if token_answer_span is None:
            token_answer_span = Span(-1, -1)

        # make the metadata
        metadata = {
            "question": question,
            "modified_question": modified_question,
            "context": context,
            "offset_mapping": offset_mapping,
            "special_tokens_mask": special_tokens_mask,
            "answers": answers,
            "first_answer_start_offset": first_answer_start_offset,
            "id": qid,
            "window_index": window_index,
            "token_answer_span": token_answer_span,
            "is_impossible": is_impossible,
            "is_boolq": is_boolq,
        }
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
