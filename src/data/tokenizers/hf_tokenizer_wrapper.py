from typing import Dict, List, Any, Optional

import os

from allennlp.common import Registrable

from transformers import AutoTokenizer
from transformers import AutoConfig


class HFTokenizerWrapper(Registrable):

    default_implementation = "default"

    def __init__(
        self,
        pretrained_model: Optional[str],
        serialization_dir: Optional[str] = None,
        init_kwargs: Dict[str, Any] = {},
        call_kwargs: Dict[str, Any] = {},
    ):
        self.pretrained_model = pretrained_model

        self._init_kwargs = {
            "use_fast": True
        }  # This affects stuff, e.g. when return_overflowing_tokens=True
        self._init_kwargs.update(init_kwargs)

        self._call_kwargs = {}
        self._call_kwargs.update(call_kwargs)

        self.tokenizer = self.load(serialization_dir)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, **self._init_kwargs)

    def encode(self, text: str, text_pair: str = None, **kwargs) -> List[int]:
        call_kwargs = self._call_kwargs.copy()
        call_kwargs.update(kwargs)
        return self.tokenizer(text, text_pair, **call_kwargs)

    def tokenize(self, text: str, text_pair: str = None, **kwargs) -> List[str]:
        return self.convert_ids_to_tokens(self.encode(text, text_pair)["input_ids"])

    @property
    def convert_ids_to_tokens(self):
        return self.tokenizer.convert_ids_to_tokens

    @property
    def convert_tokens_to_ids(self):
        return self.tokenizer.convert_tokens_to_ids

    def __get_dir_name(self, pending):
        return "pending_tokenizer" if pending else "tokenizer"

    def load(self, serialization_dir: Optional[str], pending=False):
        dir_name = self.__get_dir_name(pending)
        tokenizer = None
        if serialization_dir is not None:
            tokenizer_path = os.path.join(serialization_dir, dir_name)
            if os.path.isdir(tokenizer_path):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **self._init_kwargs)
        return tokenizer

    def save(self, serialization_dir: str, pending=False):
        dir_name = self.__get_dir_name(pending)
        tokenizer_path = os.path.join(serialization_dir, dir_name)
        if not os.path.isdir(tokenizer_path):
            config = AutoConfig.from_pretrained(self.pretrained_model)
            config.save_pretrained(tokenizer_path)
            self.tokenizer.save_pretrained(tokenizer_path)


HFTokenizerWrapper.register("default")(HFTokenizerWrapper)
