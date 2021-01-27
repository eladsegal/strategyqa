from typing import Dict

import os
from overrides import overrides
import torch

from tempfile import TemporaryDirectory
import tarfile

import logging

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from transformers import AutoModelForSequenceClassification

from src.data.tokenizers.hf_tokenizer_wrapper import HFTokenizerWrapper

logger = logging.getLogger(__name__)


@Model.register("hf_classifier")
class HFClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        serialization_dir: str,
        pretrained_model: str,
        tokenizer_wrapper: HFTokenizerWrapper,
        num_labels: int,
        label_namespace: str = "labels",
        transformer_weights_path: str = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._tokenizer_wrapper = tokenizer_wrapper
        self._label_namespace = label_namespace

        pre_serialization_dir = os.environ.get("pre_serialization_dir", None)
        if pre_serialization_dir is not None:
            tokenizer_wrapper.tokenizer = tokenizer_wrapper.load(pre_serialization_dir)

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        self._accuracy = CategoricalAccuracy()

        self._classifier = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=self._num_labels, return_dict=True
        )
        self._classifier.resize_token_embeddings(len(tokenizer_wrapper.tokenizer))

        if transformer_weights_path is not None:
            with TemporaryDirectory() as tmpdirname:
                with tarfile.open(transformer_weights_path, mode="r:gz") as input_tar:
                    logger.info("Extracting model...")
                    input_tar.extractall(tmpdirname)

                model_state = torch.load(
                    os.path.join(tmpdirname, "weights.th"),
                    map_location=util.device_mapping(-1),
                )

                source_prefix = "_transformers_model."
                target_prefix = "_classifier." + self._classifier.base_model_prefix + "."
                for target_name, parameter in self.named_parameters():
                    if not target_name.startswith(target_prefix):
                        continue
                    source_name = source_prefix + target_name[len(target_prefix) :]
                    source_weights = model_state[source_name]
                    parameter.data.copy_(source_weights.data)

        initializer(self)
        self._tokenizer_wrapper.tokenizer = self._tokenizer_wrapper.load(
            serialization_dir, pending=True
        )
        self._tokenizer_wrapper.save(serialization_dir)
        self._classifier.resize_token_embeddings(len(tokenizer_wrapper.tokenizer))

    def forward(  # type: ignore
        self, tokens: Dict[str, torch.Tensor], label: torch.IntTensor = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        output_dict = self._classifier(**tokens, labels=label)

        if label is not None:
            self._accuracy(output_dict["logits"], label)

        output_dict["probs"] = torch.nn.functional.softmax(output_dict["logits"], dim=-1)
        if "qid" in next(iter(kwargs["metadata"]), {}):
            output_dict["qid"] = [metadata_entry["qid"] for metadata_entry in kwargs["metadata"]]

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            index_to_token_vocab = self.vocab.get_index_to_token_vocabulary(self._label_namespace)
            if label_idx in index_to_token_vocab:
                label_str = index_to_token_vocab.get(label_idx)
            else:
                label_str = label_idx == 1
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
