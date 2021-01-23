import json
import logging
from typing import Optional, List

from allennlp.data.fields import MetadataField, LabelField
from overrides import overrides

from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from src.data.dataset_readers.base.base_dataset_reader import BaseDatasetReader
from src.data.fields.dictionary_field import DictionaryField
from src.data.fields.labels_field import LabelsField

logger = logging.getLogger(__name__)


@DatasetReader.register("strategy_decomposition_reader")
class StrategyQADecompositionReader(BaseDatasetReader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @overrides
    def _direct_read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading the dataset:")
        logger.info("Reading file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            for item in dataset:
                instance = self._item_to_instance(item)
                if instance is not None:
                    yield instance

    def _item_to_instance(self, item):
        question: str = item["question"]
        decomposition: Optional[List[str]] = item["decomposition"] if "decomposition" in item else None

        if not self._is_training or decomposition is not None:
            instance = self.text_to_instance(question, decomposition)
            if instance is not None:
                instance["metadata"].metadata["qid"] = item["qid"]
            return instance
        return None

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        decomposition: Optional[List[str]] = None,
    ) -> Instance:
        tokenizer_wrapper = self._tokenizer_wrapper
        fields = {}
        pad_token_id = tokenizer_wrapper.tokenizer.pad_token_id

        encoded_input = tokenizer_wrapper.encode(question)
        fields["source"] = DictionaryField(
            {
                key: LabelsField(value, padding_value=pad_token_id)
                if key == "input_ids"
                else LabelsField(value)
                for key, value in encoded_input.items()
            }
        )

        if decomposition is not None:
            decomposition_str = f" {tokenizer_wrapper.tokenizer.bos_token} ".join(decomposition)
            encoded_target = tokenizer_wrapper.encode(decomposition_str)
            fields["target_ids"] = LabelsField(
                encoded_target["input_ids"], padding_value=pad_token_id
            )
        else:
            fields["target_ids"] = LabelsField(
                [],
                padding_value=pad_token_id,
            )

        fields["decoder_start_token_id"] = LabelField(
            tokenizer_wrapper.tokenizer.bos_token_id, skip_indexing=True
        )

        # make the metadata
        metadata = {
            "question": question,
        }
        if decomposition is not None:
            metadata.update({
                "gold_decomposition": decomposition
            })
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
