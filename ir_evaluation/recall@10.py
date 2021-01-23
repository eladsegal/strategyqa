import argparse
import json


def recall(relevant_paragraphs, retrieved_paragraphs):
    result = len(set(relevant_paragraphs).intersection(retrieved_paragraphs)) / len(
        relevant_paragraphs
    )
    return result


def calculate_score(args):
    with open(args.data, "r", encoding="utf8") as f:
        dataset = json.load(f)
    with open(args.retrieved_paragraphs, "r", encoding="utf8") as f:
        all_retrieved_paragraphs = json.load(f)

    instance_scores = []
    for json_obj in dataset:
        evidence_per_annotator = []
        for annotator in json_obj["evidence"]:
            evidence_per_annotator.append(
                set(
                    evidence_id
                    for step in annotator
                    for x in step
                    if isinstance(x, list)
                    for evidence_id in x
                )
            )
        retrieved_paragraphs = all_retrieved_paragraphs[json_obj["qid"]][: args.retrieval_limit]

        score_per_annotator = []
        for evidence in evidence_per_annotator:
            score = recall(evidence, retrieved_paragraphs) if len(evidence) > 0 else 0
            score_per_annotator.append(score)

        annotator_maximum = max(score_per_annotator)
        instance_scores.append(annotator_maximum)

    if len(instance_scores) > 0:
        score = sum(instance_scores) / len(instance_scores)
    else:
        score = None

    output = {"#instances": len(instance_scores), f"Recall@{args.retrieval_limit}": score}
    for k, v in output.items():
        print(f"{k}: {v}")
    if args.output_file is not None:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(
                output,
                f,
                ensure_ascii=False,
                indent=4,
            )


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data", type=str, required=True)
    parse.add_argument("-r", "--retrieved-paragraphs", type=str, required=True)
    parse.add_argument("--output-file", type=str)
    parse.add_argument("--retrieval-limit", type=int, default=10)
    args = parse.parse_args()
    return calculate_score(args)


if __name__ == "__main__":
    main()
