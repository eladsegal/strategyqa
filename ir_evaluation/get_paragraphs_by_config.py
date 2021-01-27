import argparse
import json
import os

from allennlp.common.params import Params
from allennlp.data import DatasetReader
from allennlp.common.util import import_module_and_submodules


def get_paragraphs(args):
    import_module_and_submodules("src")
    os.environ["weights"] = ""

    overrides_dict = {}
    ext_vars = {}
    params = Params.from_file(args.config_file, json.dumps(overrides_dict), ext_vars)
    dataset_reader = DatasetReader.from_params(
        params.get("validation_dataset_reader", "dataset_reader")
    )

    data_path = args.data
    if data_path is None:
        data_path = params.get("validation_data_path", None)
    assert data_path is not None, "--data is required"

    retrieved_paragraphs = {}
    for instance in dataset_reader.read(data_path):
        paragraphs_objs = instance["metadata"]["paragraphs"]
        if paragraphs_objs is not None:
            retrieved_paragraphs[instance["metadata"]["qid"]] = [
                p["evidence_id"] for p in paragraphs_objs
            ]
        else:
            retrieved_paragraphs[instance["metadata"]["qid"]] = []

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(retrieved_paragraphs, f, ensure_ascii=False, indent=4)


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-c", "--config-file", type=str, required=True)
    parse.add_argument("--output-file", type=str, required=True)
    parse.add_argument("--data", type=str)
    args = parse.parse_args()
    return get_paragraphs(args)


if __name__ == "__main__":
    main()
