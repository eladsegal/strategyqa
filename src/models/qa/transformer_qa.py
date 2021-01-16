import logging
from typing import Any, Dict, List, Optional

import torch

from allennlp.models.model import Model
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from torch.nn import CrossEntropyLoss

from allennlp_models.rc.models.utils import (
    get_best_span,
    replace_masked_values_with_big_negative_number,
)
from src.metrics.squad2_em_and_f1 import Squad2EmAndF1

from src.data.tokenizers.tokens_interpreter import TokensInterpreter
from src.data.types import Span

from transformers import AutoModelForQuestionAnswering

from src.data.tokenizers.hf_tokenizer_wrapper import HFTokenizerWrapper

logger = logging.getLogger(__name__)


@Model.register("transformer_qa_v2")
class TransformerQA(Model):
    def __init__(
        self,
        serialization_dir: str,
        pretrained_model: str,
        tokenizer_wrapper: HFTokenizerWrapper,
        enable_no_answer: bool = False,
        force_yes_no: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer_wrapper = tokenizer_wrapper

        self._enable_no_answer = enable_no_answer
        self.force_yes_no = force_yes_no

        self._qa_model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model, return_dict=True
        )
        self._qa_model.resize_token_embeddings(len(tokenizer_wrapper.tokenizer))

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._boolq_accuracy = Squad2EmAndF1()
        self._per_instance_metrics = Squad2EmAndF1()

        # Initializer placeholder
        self._tokenizer_wrapper.tokenizer = self._tokenizer_wrapper.load(
            serialization_dir, pending=True
        )
        self._tokenizer_wrapper.save(serialization_dir)
        self._qa_model.resize_token_embeddings(len(tokenizer_wrapper.tokenizer))

    def forward(  # type: ignore
        self,
        question_with_context: Dict[str, Dict[str, torch.LongTensor]],
        context_span: torch.IntTensor,
        yes_no_span: torch.IntTensor = None,
        answer_span: Optional[torch.IntTensor] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        question_with_context : `Dict[str, torch.LongTensor]`
            From a ``TextField``. The model assumes that this text field contains the context followed by the
            question. It further assumes that the tokens have type ids set such that any token that can be part of
            the answer (i.e., tokens from the context) has type id 0, and any other token (including [CLS] and
            [SEP]) has type id 1.
        context_span : `torch.IntTensor`
            From a ``SpanField``. This marks the span of word pieces in ``question`` from which answers can come.
        answer_span : `torch.IntTensor`, optional
            From a ``SpanField``. This is the thing we are trying to predict - the span of text that marks the
            answer. If given, we compute a loss that gets included in the output directory.
        metadata : `List[Dict[str, Any]]`, optional
            If present, this should contain the question id, and the original texts of context, question, tokenized
            version of both, and a list of possible answers. The length of the ``metadata`` list should be the
            batch size, and each dictionary should have the keys ``id``, ``question``, ``context``,
            ``question_tokens``, ``context_tokens``, and ``answers``.

        # Returns

        An output dictionary consisting of:

        span_start_logits : `torch.FloatTensor`
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : `torch.FloatTensor`
            The result of `softmax(span_start_logits)`.
        span_end_logits : `torch.FloatTensor`
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : `torch.FloatTensor`
            The result of ``softmax(span_end_logits)``.
        best_span : `torch.IntTensor`
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        best_span_scores : `torch.FloatTensor`
            The score for each of the best spans.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        best_span_str : `List[str]`
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        outputs = self._qa_model(**question_with_context)
        span_start_logits = outputs["start_logits"]
        span_end_logits = outputs["end_logits"]

        with torch.no_grad():
            possible_answer_mask = torch.zeros_like(
                question_with_context["input_ids"],
                dtype=torch.bool,
            )
            if not self.force_yes_no:
                for i, (start, end) in enumerate(context_span):
                    if start != -1 and end != -1:
                        possible_answer_mask[i, start : end + 1] = True

            if yes_no_span is not None:
                for i, (start, end) in enumerate(yes_no_span):
                    if start != -1 and end != -1:
                        possible_answer_mask[i, start : end + 1] = True

            for i in range(len(possible_answer_mask)):
                assert any(possible_answer_mask[i])

            # Replace the masked values with a very negative constant.
            context_masked_span_start_logits = replace_masked_values_with_big_negative_number(
                span_start_logits, possible_answer_mask
            )
            context_masked_span_end_logits = replace_masked_values_with_big_negative_number(
                span_end_logits, possible_answer_mask
            )
            best_spans = get_best_span(
                context_masked_span_start_logits, context_masked_span_end_logits
            )
            best_span_scores = torch.gather(
                context_masked_span_start_logits, 1, best_spans[:, 0].unsqueeze(1)
            ) + torch.gather(context_masked_span_end_logits, 1, best_spans[:, 1].unsqueeze(1))
            best_span_scores = best_span_scores.squeeze(1)

            output_dict = {
                "best_span": best_spans,
                "best_span_scores": best_span_scores,
                "yes_scores": span_start_logits[:, yes_no_span[:, 0]]
                + span_end_logits[:, yes_no_span[:, 0]],
                "no_scores": span_start_logits[:, yes_no_span[:, 1]]
                + span_end_logits[:, yes_no_span[:, 1]],
            }
            if self._enable_no_answer:
                no_answer_scores = span_start_logits[:, 0] + span_end_logits[:, 0]
                output_dict.update({"no_answer_scores": no_answer_scores})

        # Compute metrics and set loss
        if answer_span is not None:
            span_start = answer_span[:, 0]
            span_end = answer_span[:, 1]
            span_mask = span_start != -1
            if self._enable_no_answer:
                span_mask &= span_start != 0

            self._span_accuracy(
                best_spans, answer_span, span_mask.unsqueeze(-1).expand_as(best_spans)
            )

            self._span_start_accuracy(context_masked_span_start_logits, span_start, span_mask)
            self._span_end_accuracy(context_masked_span_end_logits, span_end, span_mask)

            if self._enable_no_answer:
                possible_answer_mask[:, 0] = True
            # Replace the masked values with a very negative constant.
            masked_span_start_logits = replace_masked_values_with_big_negative_number(
                span_start_logits, possible_answer_mask
            )
            masked_span_end_logits = replace_masked_values_with_big_negative_number(
                span_end_logits, possible_answer_mask
            )

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(masked_span_start_logits, span_start)
            end_loss = loss_fct(masked_span_end_logits, span_end)
            total_loss = (start_loss + end_loss) / 2
            output_dict["loss"] = total_loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            best_spans = best_spans.detach().cpu().numpy()

            output_dict["best_span_str"] = []
            for i, (metadata_entry, best_span) in enumerate(zip(metadata, best_spans)):
                best_span_string = TokensInterpreter.extract_span_string_from_origin_texts(
                    Span(*best_span),
                    [metadata_entry["modified_question"], metadata_entry["context"]],
                    metadata_entry["offset_mapping"],
                    metadata_entry["special_tokens_mask"],
                )

                if self.force_yes_no:
                    if output_dict["yes_scores"][i].item() > output_dict["no_scores"][i].item():
                        overriding_best_span_string = "yes"
                    else:
                        overriding_best_span_string = "no"
                    if overriding_best_span_string != best_span_string:
                        best_span_string = overriding_best_span_string

                output_dict["best_span_str"].append(best_span_string)

                answers = metadata_entry.get("answers")
                if answers is not None and len(answers) > 0:
                    if self._enable_no_answer:
                        final_pred = (
                            best_span_string if best_span_scores[i] > no_answer_scores[i] else ""
                        )
                    else:
                        final_pred = best_span_string

                    if metadata_entry["is_boolq"]:
                        self._boolq_accuracy(final_pred, answers)
                    else:
                        self._per_instance_metrics(final_pred, answers)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
            "boolq_acc": self._boolq_accuracy.get_metric(reset)["em"],
        }
        metrics.update(self._per_instance_metrics.get_metric(reset))
        return metrics

    default_predictor = "transformer_qa_v2"
