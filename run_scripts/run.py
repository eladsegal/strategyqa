import sys
import argparse
import json

from allennlp.commands import main as allennlp_main
from allennlp.common.params import with_fallback

import logging


def run(args):
    overrides_dict = {}

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    sys.argv = sys.argv[sys.argv.index("allennlp") :]

    overrides_index = -1
    try:
        overrides_index = sys.argv.index("-o") + 1
    except:  # noqa: E722
        try:
            overrides_index = sys.argv.index("--overrides") + 1
        except:  # noqa: E722
            pass

    if overrides_index != -1:
        overrides = sys.argv[overrides_index]
        if args.hard_overrides:
            new_overrides_dict = json.loads(overrides)
            new_overrides_dict.update(overrides_dict)
            sys.argv[overrides_index] = json.dumps(new_overrides_dict)
        else:
            sys.argv[overrides_index] = json.dumps(
                with_fallback(preferred=overrides_dict, fallback=json.loads(overrides))
            )
    else:
        sys.argv += ["-o", json.dumps(overrides_dict)]

    print(sys.argv)
    print(" ".join(sys.argv))
    allennlp_main()


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--debug", action="store_true", default=False)
    parse.add_argument("--hard-overrides", action="store_false", default=True)
    args, _ = parse.parse_known_args()
    run(args)


if __name__ == "__main__":
    main()
