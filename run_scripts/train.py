import argparse
import json
import os
import sys

from run import main as run_main


def run(args):
    if args.weights is not None:
        os.environ["weights"] = args.weights

    overrides_dict = {}

    if args.debug:
        overrides_dict.update(
            {
                "dataset_reader": {"max_instances": 50, "pickle": None},
                "validation_dataset_reader": {"max_instances": 50, "pickle": None},
            }
        )

    cuda_device = args.gpu
    if cuda_device != "":
        cuda_device = eval(cuda_device)
        if isinstance(cuda_device, int):
            overrides_dict["trainer"] = {"cuda_device": cuda_device}

        elif isinstance(cuda_device, list):
            overrides_dict["distributed"] = {"cuda_devices": cuda_device}
        else:
            raise ValueError("cuda_devices must be a list of a int")

    if args.serialization_dir is None:
        index = 0
        basename = os.path.splitext(os.path.basename(args.config_file))[0]
        base_dirname = os.path.basename(os.path.dirname(args.config_file))
        base_serialization_dir = f"experiments/{base_dirname}_{basename}"
        while True:
            serialization_dir = f"{base_serialization_dir}_{index}"
            if os.path.exists(serialization_dir):
                index += 1
            else:
                break
        if args.force and index > 0:
            index -= 1
            serialization_dir = f"{base_serialization_dir}_{index}"
    else:
        serialization_dir = args.serialization_dir

    overrides_dict.update(json.loads(args.overrides))
    overrides = json.dumps(overrides_dict)

    sys.argv = (
        ["run.py"]
        + (["--debug"] if args.debug else [])
        + ["allennlp"]
        + ["train"]
        + [args.config_file]
        + ["-s", serialization_dir]
        + ["--include-package", "src"]
        + ["-o", overrides]
        + (["--force"] if args.force else [])
        + (["--recover"] if args.recover else [])
    )

    print(sys.argv)
    print(" ".join(sys.argv))
    run_main()

    return serialization_dir


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--debug", action="store_true", default=False)
    parse.add_argument("--force", action="store_true", default=False)
    parse.add_argument("--recover", action="store_true", default=False)
    parse.add_argument("-g", "--gpu", type=str, default="", help="CUDA device")
    parse.add_argument(
        "-c",
        "--config-file",
        type=str,
        help="Config file, must for train",
        required=True,
    )
    parse.add_argument(
        "-s",
        "--serialization-dir",
        type=str,
        help="Serialization dir. If empty, gets default name based on the config",
    )
    parse.add_argument("-o", "--overrides", type=str, default="{}", help="Overrides")
    parse.add_argument("-w", "--weights", type=str, help="Weights to initialize from")
    args = parse.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
