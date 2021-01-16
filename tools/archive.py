from allennlp.models.archival import archive_model
import os
import argparse


def main(args):
    os.makedirs(
        os.path.dirname(args.output_file) if os.path.dirname(args.output_file) != "" else ".",
        exist_ok=True,
    )
    if args.weights_file is None:
        archive_model(args.model_dir, archive_path=args.output_file)
    else:
        archive_model(args.model_dir, weights=args.weights_file, archive_path=args.output_file)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("model_dir", type=str, help="the serialization dir")
    parse.add_argument("output_file", type=str, help="the path to create the output in")
    parse.add_argument(
        "--weights-file", type=str, default=None, help="the name of the weights to use"
    )
    args = parse.parse_args()

    main(args)
