import argparse
import json
import os
import sys
import tempfile
import tarfile

from allennlp.common.params import Params
from allennlp.models.archival import CONFIG_NAME

from run import main as run_main


def run(args):
    overrides_dict = {}
    if args.debug:
        overrides_dict.update(
            {
                "validation_dataset_reader": {"max_instances": 50, "pickle": None},
            }
        )

    overrides_dict.update(json.loads(args.overrides))
    overrides = json.dumps(overrides_dict)

    data = None
    if args.data is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(args.model, "r:gz") as archive:
                archive.extract(CONFIG_NAME, temp_dir)
            config = Params.from_file(os.path.join(temp_dir, CONFIG_NAME))
            data = config.get("validation_data_path", None)
    else:
        data = args.data
    assert data is not None, "--data is required"

    output_file = args.output_file
    if output_file is None:
        base_dirname = os.path.dirname(args.model)
        data_name = os.path.splitext(os.path.basename(data))[0]
        output_name = args.output_name
        if output_name is None:
            output_name = f"preds_{data_name}.jsonl"
        output_file = os.path.join(base_dirname, output_name)

    sys.argv = (
        ["run.py"]
        + (["--debug"] if args.debug else [])
        + ["allennlp"]
        + ["predict"]
        + (["--extend-vocab"] if args.extend_vocab else [])
        + ["--include-package", "src"]
        + ["--cuda-device", args.gpu]
        + ["--output-file", output_file]
        + (["--batch-size", args.batch_size] if args.batch_size else [])
        + (["--weights-file", args.weights_file] if args.weights_file else [])
        + ["--use-dataset-reader"]
        + [args.model]
        + [data]
        + ["-o", overrides]
        + (["--silent"] if args.silent else [])
    )

    print(sys.argv)
    print(" ".join(sys.argv))
    run_main()


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--debug", action="store_true", default=False)
    parse.add_argument("-g", "--gpu", type=str, default="-1", help="CUDA device")
    parse.add_argument("--output-file", type=str)
    parse.add_argument("--output-name", type=str)
    parse.add_argument("--extend-vocab", action="store_true", default=False)
    parse.add_argument("--silent", action="store_true", default=False)
    parse.add_argument("--batch-size", type=str, help="weights file path")
    parse.add_argument("--weights-file", type=str, help="weights file path")
    parse.add_argument("--model", type=str, help="model.tar.gz", required=True)
    parse.add_argument("--data", type=str, help="data path")
    parse.add_argument("-o", "--overrides", type=str, default="{}", help="Overrides")
    args = parse.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
