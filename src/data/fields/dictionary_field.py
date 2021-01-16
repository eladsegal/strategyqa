"""
A `TextField` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.
"""
from typing import Dict, List, Any, Optional

from overrides import overrides
import torch

from allennlp.data.fields import Field
from allennlp.data.vocabulary import Vocabulary


class DictionaryField(Field[Dict[str, torch.Tensor]]):
    """
    Can actually return Dict[str, Any], but usually Any will be torch.Tensor.
    Supports only one level nesting (dictionary within a dictionary)
    """

    __slots__ = ["field_dict", "_length"]

    def __init__(self, field_dict: Dict[str, Field], length: Optional[int] = None) -> None:
        self.field_dict = field_dict
        self._length = length

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for field in self.field_dict.values():
            field.count_vocab_items(counter)

    @overrides
    def index(self, vocab: Vocabulary):
        for field in self.field_dict.values():
            field.index(vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        field_lengths = {key: field.get_padding_lengths() for key, field in self.field_dict.items()}
        padding_lengths = {}

        for field_key, field_lengths in field_lengths.items():
            for key, field_length in field_lengths.items():
                padding_lengths[f"dict_{field_key}_{key}"] = field_length

        return padding_lengths

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        tensors = {}
        child_padding_lengths = {
            key.replace("dict_", "", 1): value
            for key, value in padding_lengths.items()
            if key.startswith("dict_")
        }
        for key, field in self.field_dict.items():
            subchild_padding_lengths = {
                padding_length_key.replace(f"{key}_", "", 1): value
                for padding_length_key, value in child_padding_lengths.items()
                if padding_length_key.startswith(f"{key}_")
            }
            tensors[key] = field.as_tensor(subchild_padding_lengths)
        return tensors

    @overrides
    def empty_field(self):
        dict_field = DictionaryField(
            {key: field.empty_field() for key, field in self.field_dict.items()}
        )
        return dict_field

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return self.batch_dicts(tensor_list)

    def batch_dicts(self, dict_list: List[Dict[str, Any]]):
        # Assuming each dict in the list have the same keys as the first one
        representative_dict = dict_list[0]
        keys = representative_dict.keys()
        batched_dicts = {}
        for key in keys:
            if isinstance(representative_dict[key], dict):
                batched_dicts[key] = self.batch_dicts([d[key] for d in dict_list])
            else:
                batched_dicts[key] = super().batch_tensors([d[key] for d in dict_list])
        return batched_dicts

    def __str__(self) -> str:
        base_string = f"DictionaryField of {len(self.field_dict)} keys : \n"
        return " ".join(
            [base_string] + [f"\t{key}: {field} \n" for key, field in self.field_dict.items()]
        )

    def __getitem__(self, key: str) -> Field:
        return self.field_dict[key]

    def __len__(self) -> int:
        return self._length if self._length is not None else len(self.field_dict)
