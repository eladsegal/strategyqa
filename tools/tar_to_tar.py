import argparse
import os

from tempfile import TemporaryDirectory
import tarfile

from allennlp.common.params import Params


def main(args):
    with TemporaryDirectory() as tmpdirname:
        with tarfile.open(args.src_model, mode="r:gz") as input_tar:
            print("Extracting model...")
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(input_tar, tmpdirname)

        Params.from_file(args.config).to_file(os.path.join(tmpdirname, "config.json"))

        os.makedirs(
            os.path.dirname(args.dest_model) if os.path.dirname(args.dest_model) != "" else ".",
            exist_ok=True,
        )
        with tarfile.open(args.dest_model, "w:gz") as output_tar:
            print("Archiving model...")
            output_tar.add(tmpdirname, arcname="")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("src_model", type=str, help="Source model.tar.gz to modify")
    parse.add_argument("config", type=str, help="Configuration file to use")
    parse.add_argument("dest_model", type=str, help="Output path")
    args = parse.parse_args()

    main(args)
