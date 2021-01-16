import torch
import json
import logging
import time
from copy import deepcopy
from typing import Optional

from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.training.metrics import BooleanAccuracy

from tqdm import tqdm

from src.data.dataset_readers.strategy_qa_reader import StrategyQAReader
from src.models.iterative.reference_utils import (
    fill_in_references,
    get_reachability,
)

logger = logging.getLogger(__name__)


def main(
    gpu: int,
    qa_model_path: str,
    paragraphs_source: str,
    generated_decompositions_paths: Optional[str],
    data: str,
    output_predictions_file: str,
    output_metrics_file: str,
    overrides="{}",
):
    import_module_and_submodules("src")

    overrides_dict = {}
    overrides_dict.update(json.loads(overrides))
    archive = load_archive(qa_model_path, cuda_device=gpu, overrides=json.dumps(overrides_dict))
    predictor = Predictor.from_archive(archive)

    dataset_reader = StrategyQAReader(
        paragraphs_source=paragraphs_source,
        generated_decompositions_paths=generated_decompositions_paths,
    )

    accuracy = BooleanAccuracy()
    last_logged_scores_time = time.monotonic()

    logger.info("Reading the dataset:")
    logger.info("Reading file at %s", data)
    dataset = None
    with open(data, mode="r", encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)

    output_dataset = []
    for json_obj in tqdm(dataset):
        item = dataset_reader.json_to_item(json_obj)
        decomposition = item["decomposition"]
        generated_decomposition = item["generated_decomposition"]
        gold_answer = torch.tensor(item["answer"]).view((1,))

        used_decomposition = deepcopy(
            generated_decomposition
            if "generated_decomposition" in paragraphs_source
            else decomposition
        )

        # Per instance:
        # Until the final step has an answer, find in each iteration
        # all of the steps that are required to answer the last step (including by proxy)
        # and don't have references in them.
        # If it is not possible, return a score of zero for the instance.
        # If it is possible, retrieve paragraphs for these steps,
        # and then pass the step and the paragraphs for it to be answered by the model.
        # Replace the answer in all of the steps that has a reference for it.

        step_answers = [None for i in range(len(used_decomposition))]
        while True:
            reachability = get_reachability([step["question"] for step in used_decomposition])
            if reachability is None:
                break

            if step_answers[-1] is not None:
                break

            indices_of_interest = []
            if (sum(reachability[-1])) != 0:
                for i, reachable in enumerate(reachability[-1]):
                    if reachable > 0 and sum(reachability[i]) == 0:
                        indices_of_interest.append(i)
            else:
                indices_of_interest.append(len(step_answers) - 1)

            paragraphs = dataset_reader.get_paragraphs(
                decomposition=[used_decomposition[i] for i in indices_of_interest],
            )
            if paragraphs is not None:
                paragraphs_per_step_of_interest = paragraphs["per_step"]
            else:
                paragraphs_per_step_of_interest = [[{"content": " "}] for i in indices_of_interest]

            for i in indices_of_interest:
                step_answers[i] = get_answer(
                    predictor=predictor,
                    question=used_decomposition[i]["question"],
                    paragraphs=paragraphs_per_step_of_interest[indices_of_interest.index(i)],
                    force_yes_no=i == len(step_answers) - 1,
                )  # Return the best non-empty answer

            for i, step in enumerate(used_decomposition):
                used_decomposition[i]["question"] = fill_in_references(
                    step["question"], step_answers
                )

        predicted_answer_str = step_answers[-1].lower() if step_answers[-1] is not None else None
        if predicted_answer_str == "yes" or predicted_answer_str == "no":
            # Valid answer, the metric should be updated accordingly
            predicted_answer = torch.tensor(predicted_answer_str == "yes").view((1,))
            accuracy(predicted_answer, gold_answer)
        else:
            # Invalid answer, the metric should be updated with a mistake
            accuracy(not gold_answer, gold_answer)

        if time.monotonic() - last_logged_scores_time > 3:
            metrics_dict = {"accuracy": accuracy.get_metric()}
            logger.info(json.dumps(metrics_dict))
            last_logged_scores_time = time.monotonic()

        output_json_obj = deepcopy(json_obj)
        output_json_obj["decomposition"] = [step["question"] for step in used_decomposition]
        output_json_obj["step_answers"] = step_answers
        output_dataset.append(output_json_obj)

    if output_predictions_file is not None:
        with open(output_predictions_file, "w", encoding="utf-8") as f:
            json.dump(output_dataset, f, ensure_ascii=False, indent=4)

    metrics_dict = {"accuracy": accuracy.get_metric(reset=True)}
    if output_metrics_file is None:
        print(json.dumps(metrics_dict))
    else:
        with open(output_metrics_file, "w", encoding="utf-8") as f:
            json.dump(
                metrics_dict,
                f,
                ensure_ascii=False,
                indent=4,
            )


def get_answer(predictor, question, paragraphs, force_yes_no):
    max_score = float("-inf")
    answer = None
    for paragraph in paragraphs:
        content = paragraph["content"]
        instances = predictor._batch_json_to_instances(
            [
                {
                    "context": content,
                    "question": question,
                }
            ]
        )
        result = predictor.predict_batch_instance(
            instances, allow_null=False, force_yes_no=force_yes_no
        )[0]
        if max_score < result["best_span_scores"]:
            max_score = result["best_span_scores"]
            answer = result["best_span_str"]
    return answer


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parse = argparse.ArgumentParser(description="Iterative model for StrategyQA")
    parse.add_argument("-g", "--gpu", type=int, default="", help="CUDA device")
    parse.add_argument("--qa-model-path", type=str, required=True)
    parse.add_argument("--paragraphs-source", type=str)
    parse.add_argument("--generated-decompositions-paths", type=str, nargs="*")
    parse.add_argument("--data", type=str)
    parse.add_argument("--output-predictions-file", type=str)
    parse.add_argument("--output-metrics-file", type=str)
    parse.add_argument("-o", "--overrides", type=str, default="{}", help="Overrides")
    args = parse.parse_args()

    main(**vars(args))
