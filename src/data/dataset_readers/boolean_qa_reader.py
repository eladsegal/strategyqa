import json
import logging
from typing import Optional

from allennlp.data.fields import MetadataField, LabelField
from overrides import overrides

from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from src.data.dataset_readers.base.base_dataset_reader import BaseDatasetReader
from src.data.fields.dictionary_field import DictionaryField
from src.data.fields.labels_field import LabelsField

logger = logging.getLogger(__name__)


@DatasetReader.register("boolean_qa_reader")
class BooleanQAReader(BaseDatasetReader):
    def __init__(
        self,
        with_context: bool = True,
        context_key: str = "context",
        answer_key: str = "answer",
        is_twenty_questions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._with_context = with_context
        self._context_key = context_key
        self._answer_key = answer_key
        self._is_twenty_questions = is_twenty_questions

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading the dataset:")
        logger.info("Reading file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            for line in dataset_file:
                item = json.loads(line)
                question = item["question"]

                if self._is_twenty_questions:
                    subject = item["subject"]
                    original_question = question
                    question = question.replace(" it ", " " + subject + " ")
                    question = question.replace(" it?", " " + subject)
                    if question.endswith(" it"):
                        question = question[: -len("it")] + subject
                    if question.startswith("it "):
                        question = subject + question[len("it")]
                    if original_question == question:
                        continue
                    question = question.replace("?", "")
                    question = question.replace("\t", "")

                question: str = question if question.endswith("?") else question + "?"
                context: str = item[self._context_key] if self._context_key in item else None
                answer: Optional[bool] = (
                    item[self._answer_key] if self._answer_key in item else None
                )

                if not self._is_training or answer is not None:
                    instance = self.text_to_instance(
                        question,
                        context,
                        answer,
                    )
                    yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        context: Optional[str] = None,
        answer: Optional[bool] = None,
    ) -> Instance:
        tokenizer_wrapper = self._tokenizer_wrapper
        fields = {}
        pad_token_id = tokenizer_wrapper.tokenizer.pad_token_id

        if "?" not in question:
            question += "?"

        if self._with_context:
            if context is None:
                context = ""
            encoded_input = tokenizer_wrapper.encode(context, question)
        else:
            encoded_input = tokenizer_wrapper.encode(question)

        encoded_input_fields = {
            key: LabelsField(value, padding_value=pad_token_id)
            if key == "input_ids"
            else LabelsField(value)
            for key, value in encoded_input.items()
        }
        fields["tokens"] = DictionaryField(
            encoded_input_fields, length=len(encoded_input["input_ids"])
        )

        if answer is not None:
            fields["label"] = LabelField(int(answer), skip_indexing=True)

        # make the metadata
        metadata = {
            "question": question,
            "context": context,
            "answer": answer,
            "tokenized_input": tokenizer_wrapper.convert_ids_to_tokens(encoded_input["input_ids"]),
        }
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
