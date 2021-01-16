from typing import List, Dict, Any

from allennlp.models import Model
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.predictors.predictor import Predictor


@Predictor.register("transformer_qa_v2")
class TransformerQAPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp_rc.models.TransformerQA` model, and any
    other model that takes a question and passage as input.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(TransformerQAPredictor, self).__init__(model, dataset_reader)
        self._next_qid = 1

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"context": passage, "question": question})

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        results = self.predict_batch_json([inputs])
        assert len(results) == 1
        return results[0]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raise NotImplementedError(
            "This predictor maps a question to multiple instances. "
            "Please use _json_to_instances instead."
        )

    def _json_to_instances(self, json_dict: JsonDict) -> List[Instance]:
        # We allow the passage / context to be specified with either key.
        # But we do it this way so that a 'KeyError: context' exception will be raised
        # when neither key is specified, since the 'context' key is the default and
        # the 'passage' key was only added to be compatible with the input for other
        # RC models.
        context = json_dict["passage"] if "passage" in json_dict else json_dict["context"]
        result = list(
            self._dataset_reader.make_instances(
                qid=str(self._next_qid),
                question=json_dict["question"],
                answers=[],
                context=context,
            )
        )
        self._next_qid += 1
        return result

    @overrides
    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        instances = []
        for json_dict in json_dicts:
            instances.extend(self._json_to_instances(json_dict))
        return instances

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        result = self.predict_batch_instance(instances)
        assert len(result) == len(inputs)
        return result

    @overrides
    def predict_batch_instance(
        self,
        instances: List[Instance],
        group_same_id=True,
        allow_null=True,
        force_yes_no=False,
    ) -> List[JsonDict]:
        self._model.force_yes_no = force_yes_no  # Ugly hack
        outputs = self._model.forward_on_instances(instances)

        # group outputs with the same question id
        qid_to_output: Dict[str, Dict[str, Any]] = {}
        qid_to_score_null = {}
        qid_to_null_output = {}
        for instance, output in zip(instances, outputs):
            qid = instance["metadata"]["id"]

            output["answers"] = instance["metadata"]["answers"]
            output["token_answer_span"] = instance["metadata"]["token_answer_span"]

            if group_same_id:
                output["id"] = qid
                if qid in qid_to_output:
                    old_output = qid_to_output[qid]

                    if "no_answer_scores" in old_output:
                        if output["no_answer_scores"] < qid_to_score_null[qid]:
                            qid_to_score_null[qid] = output["no_answer_scores"]
                            qid_to_null_output[qid] = output

                    if old_output["best_span_scores"] < output["best_span_scores"]:
                        qid_to_output[qid] = output
                else:
                    qid_to_output[qid] = output
                    if "no_answer_scores" in output:
                        qid_to_score_null[qid] = output["no_answer_scores"]
                        qid_to_null_output[qid] = output
            else:
                window_index = instance["metadata"]["window_index"]
                full_id = f"{qid}_{str(window_index)}" if window_index is not None else qid
                output["id"] = full_id
                qid_to_output[full_id] = output
                if "no_answer_scores" in output:
                    qid_to_score_null[qid] = output["no_answer_scores"]
                    qid_to_null_output[qid] = output

        for qid, score_null in qid_to_score_null.items():
            qid_to_output[qid]["no_answer_scores"] = score_null
            if allow_null:
                if score_null > qid_to_output[qid]["best_span_scores"]:
                    qid_to_output[qid] = qid_to_null_output[qid]
                    qid_to_output[qid]["best_span_str"] = ""
                    qid_to_output[qid]["best_span"] = (-1, -1)
                    qid_to_output[qid]["best_span_scores"] = score_null

        return [sanitize(o) for o in qid_to_output.values()]
