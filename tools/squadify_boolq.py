import argparse
import json
import hashlib


def main(args):
    squadified_dataset = {"data": []}
    with open(args.boolq_src, mode="r", encoding="utf8") as input_file:
        for line in input_file:
            item = json.loads(line)
            squadified_dataset["data"].append(
                {
                    "title": item["title"],
                    "paragraphs": [
                        {
                            "qas": [
                                {
                                    "question": item["question"],
                                    "id": hashlib.md5(
                                        (item["title"] + item["question"]).encode("utf-8")
                                    ).hexdigest(),  # pseudo-id
                                    "answers": [
                                        {
                                            "text": "yes" if item["answer"] else "no",
                                            "answer_start": -1,
                                        }
                                    ],
                                    "is_impossible": False,
                                    "is_boolq": True,
                                }
                            ],
                            "context": item["passage"],
                        }
                    ],
                }
            )

    if args.append_to is not None:
        with open(args.append_to, mode="r", encoding="utf8") as input_file:
            append_to = json.load(input_file)
        append_to["data"].extend(squadified_dataset["data"])
        squadified_dataset = append_to

    with open(args.squadified_boolq_dest, mode="w", encoding="utf8") as output_file:
        json.dump(squadified_dataset, output_file, indent=4)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("boolq_src", type=str, help="Source boolq file")
    parse.add_argument("squadified_boolq_dest", type=str, help="Output path")
    parse.add_argument("--append-to", type=str, help="")
    args = parse.parse_args()

    main(args)
