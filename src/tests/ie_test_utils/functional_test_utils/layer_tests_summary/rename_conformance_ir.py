# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from genericpath import isfile
from openvino.runtime import Core
from argparse import ArgumentParser
from pathlib import Path
from os import path
from shutil import copytree
from hashlib import sha256
from utils import utils

XML_EXTENSION = ".xml"
BIN_EXTENSION = ".bin"
META_EXTENSION = ".meta"

logger = utils.get_logger('Rename Conformance IRs using hash')


def parse_arguments():
    parser = ArgumentParser()

    in_dir_help = "Path/s to input directory"
    parser.add_argument("--input_dir", help=in_dir_help, nargs="*", required=True)

    return parser.parse_args()


def create_hash(in_dir_path: Path):
    core = Core()
    models = in_dir_path.rglob("*.xml")
    for model_path in models:
        if not model_path.is_file:
            logger.error(f"File {model_path} is not exist!")
            exit(-1)
        str_to_hash = str()
        model = core.read_model(model_path)
        for input in model.inputs:
            str_to_hash += str(len(input.partial_shape)) + str(input.element_type) + str(input.node.type_info)
        for node in model.get_ordered_ops():
            str_to_hash += str(node.type_info)
        for output in model.outputs:      
            str_to_hash += str(len(output.partial_shape)) + str(output.element_type) + str(output.node.type_info)

        old_name = model_path
        new_name = model_path.name[:model_path.name.find('_') + 1] + str(sha256(str_to_hash.encode('utf-8')).hexdigest())

        bin_path = model_path.with_suffix(BIN_EXTENSION)
        meta_path = model_path.with_suffix(META_EXTENSION)

        model_path.rename(Path(model_path.parent, new_name + XML_EXTENSION))
        meta_path.rename(Path(meta_path.parent, new_name + META_EXTENSION))
        bin_path.rename(Path(bin_path.parent, new_name + BIN_EXTENSION))

        logger.info(f"{old_name} -> {model_path}")

if __name__=="__main__":
    args = parse_arguments()
    for in_dir in args.input_dir:
        logger.info(f"Starting to rename models in {in_dir}")
        create_hash(Path(in_dir))
    logger.info("The run is successfully completed")


