#!/usr/bin/env python3
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Sample usage:
#   ./scripts/init_corpus.py ./paddle_layer_models/**/*.pdmodel --join pdiparams
#   mkdir -p corpus && find ./paddle_layer_models/ -name "*.fuzz" -exec cp \{\} .//import_paddle-corpus \;

import argparse
import glob
import os
from pathlib import Path
import shutil
import sys


def globber(paths):
    """Generator extending paths with wildcards"""
    for path in paths:
        # XXX: use non-public `has_magic` here as we'd like to differentiate between glob and normal paths
        #  i.e. in the case when user specifies "normal" but non-existing path - we'd like to handle it by ourselves
        if glob.has_magic(path):
            for resolved in glob.iglob(path, recursive=True):
                yield resolved
        else:
            yield path


def main():
    """ Main entrypoint """
    parser = argparse.ArgumentParser(
        description="Join multiple files of the same name to a single *.fuzz file"
    )
    parser.add_argument("input", nargs="+", help="A file to add to the corpus")
    parser.add_argument(
        "--join",
        help="Colon separated list of file extensions to concatenate to corpus entry",
    )
    args = parser.parse_args()

    for input in globber(args.input):
        base = os.path.splitext(input)[0]
        output = f"{base}.fuzz"
        shutil.copyfile(input, output)
        if args.join:
            with open(output, "ab") as output_file:
                for join in args.join.split(":"):
                    join = f"{base}.{join}"
                    if os.path.isfile(join):
                        with open(join, "rb") as join_file:
                            output_file.write(bytes("FUZZ_NEXT_FIELD", "utf-8"))
                            output_file.write(join_file.read())


if __name__ == "__main__":
    sys.exit(main())
