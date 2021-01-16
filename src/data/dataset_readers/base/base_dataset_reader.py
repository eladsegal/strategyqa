import os
import logging
from typing import Any, Dict
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from src.data.tokenizers.hf_tokenizer_wrapper import HFTokenizerWrapper

from src.data.dataset_readers.utils.pickle_utils import (
    is_pickle_dict_valid,
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)


class BaseDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer_wrapper: HFTokenizerWrapper = None,
        save_tokenizer: bool = False,
        is_training: bool = False,
        serialization_dir: str = None,
        pickle: Dict[str, Any] = {"action": None},
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._tokenizer_wrapper = tokenizer_wrapper
        self._is_training = is_training
        self._serialization_dir = serialization_dir

        additional_special_tokens = self.additional_special_tokens = set()

        if tokenizer_wrapper is not None:
            pre_serialization_dir = os.environ.get("pre_serialization_dir", None)
            if pre_serialization_dir is not None:
                tokenizer_wrapper.tokenizer = tokenizer_wrapper.load(pre_serialization_dir)

        self._reader_specific_init()

        if tokenizer_wrapper is not None:
            if len(additional_special_tokens) > 0:
                tokenizer_wrapper.tokenizer.add_special_tokens(
                    {"additional_special_tokens": sorted(additional_special_tokens)}
                )

            # Save tokenizer
            if save_tokenizer and serialization_dir is not None:
                tokenizer_wrapper.save(serialization_dir, pending=True)

        self._pickle = pickle
        if not is_pickle_dict_valid(self._pickle):
            self._pickle = {"action": None}

    def _reader_specific_init(self):
        pass

    @overrides
    def _read(self, file_path: str):
        if not self.lazy and self._pickle["action"] == "load":
            # Try to load the data, if it fails then read it from scratch and save it
            logger.info("Trying to read the dataset from pickle")
            loaded_pkl = load_pkl(self._pickle, self._is_training)
            if loaded_pkl is not None:
                for instance in loaded_pkl:
                    yield instance
                return
            else:
                logger.info("Pickle file does not exist")
                self._pickle["action"] = "save"

        logger.info("Reading the dataset from scratch")
        instances_list = []
        for instance in self._direct_read(file_path):
            if not self.lazy:
                instances_list.append(instance)

                if len(instances_list) == self.max_instances:
                    if (
                        self._pickle["action"] == "save"
                        and self._pickle["save_even_when_max_instances"]
                    ):
                        save_pkl(instances_list, self._pickle, self._is_training)
                        self._pickle["action"] = None
            yield instance

        if not self.lazy and self._pickle["action"] == "save":
            save_pkl(instances_list, self._pickle, self._is_training)

    def _direct_read(self, file_path: str):
        raise NotImplementedError
