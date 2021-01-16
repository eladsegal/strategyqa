import os
import sys
import argparse
import wget
from train import main as train_main


def run(args):
    # Download DROP_TASE_IO_SSE.tar.gz (trained from eladsegal/tag-based-multi-span-extraction)
    os.makedirs("downloaded", exist_ok=True)
    drop_weights_path = "downloaded/DROP_TASE_IO_SSE.tar.gz"
    if os.path.isfile(drop_weights_path):
        os.remove(drop_weights_path)
    wget.download(
        "https://storage.googleapis.com/ai2i/strategyqa/models/DROP_TASE_IO_SSE.tar.gz",
        drop_weights_path,
    )

    # Train a QA model on twentyquestions
    os.environ["DROP_WEIGHTS"] = drop_weights_path
    sys.argv = (
        ["train.py"]
        + (["-g", args.gpu] if args.gpu else [])
        + ["-c", "configs/for_STAR/1_twentyquestions.jsonnet"]
    )
    print(" ".join(sys.argv))
    twentyquestions_serialization_dir = train_main()

    # Continue to train the model on BoolQ
    sys.argv = (
        ["train.py"]
        + (["-g", args.gpu] if args.gpu else [])
        + ["-c", "configs/for_STAR/2_boolq.jsonnet"]
        + ["-s", args.serialization_dir]
        + ["-w", os.path.join(twentyquestions_serialization_dir, "model.tar.gz")]
    )
    print(" ".join(sys.argv))
    return train_main()


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-g", "--gpu", type=str, default="", help="CUDA device")
    parse.add_argument(
        "-s",
        "--serialization-dir",
        type=str,
        required=True,
        help="Serialization dir. If empty, gets default name based on the config",
    )
    args = parse.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
