import argparse
import json
import os

from allennlp.common.params import Params
from allennlp.data import DatasetReader
from allennlp.common.util import import_module_and_submodules

from src.data.dataset_readers.strategy_qa_reader import QUERIES_CACHE_PATH


def get_paragraphs(args):
    import_module_and_submodules("src")
    os.environ["weights"] = ""

    if os.path.exists(QUERIES_CACHE_PATH):
        with open(QUERIES_CACHE_PATH, "r", encoding="utf8") as f:
            queries_cache = json.load(f)
    else:
        queries_cache = {}
    size_before = len(queries_cache)

    cleaned_queries_cache = {}

    combinations = {
        "configs/strategy_qa/3_STAR_IR-Q.jsonnet": [
            "data/strategyqa/train.json",
            "data/strategyqa/dev.json",
        ],
        "configs/strategy_qa/4_STAR_IR-D.jsonnet": ["data/strategyqa/dev.json"],
        "configs/strategy_qa/5_STAR_IR-ORA-D.jsonnet": [
            "data/strategyqa/train.json",
            "data/strategyqa/dev.json",
        ],
    }

    for config_file in combinations:
        for data_path in combinations[config_file]:
            overrides_dict = {}
            ext_vars = {}
            params = Params.from_file(config_file, json.dumps(overrides_dict), ext_vars)
            dataset_reader = DatasetReader.from_params(
                params.get("validation_dataset_reader", "dataset_reader")
            )

            for instance in dataset_reader.read(data_path):
                queries = instance["metadata"]["queries"]
                for query in queries:
                    if query in dataset_reader._queries_cache:
                        cleaned_queries_cache[query] = dataset_reader._queries_cache[query]

    print(f"Size before: {size_before}")
    print(f"Size after: {len(cleaned_queries_cache)}")
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(cleaned_queries_cache, f)


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--output-file", type=str, required=True)
    args = parse.parse_args()
    return get_paragraphs(args)


if __name__ == "__main__":
    main()
