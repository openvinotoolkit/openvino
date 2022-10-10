# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET

from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
from utils import utils

from openvino.runtime import Core

XML_EXTENSION = ".xml"
BIN_EXTENSION = ".bin"
META_EXTENSION = ".meta"

logger = utils.get_logger('Rename Conformance IRs using hash')


def parse_arguments():
    parser = ArgumentParser()

    in_dir_help = "Path/s to input directory"
    parser.add_argument("--input_dir", help=in_dir_help, nargs="*", required=True)

    return parser.parse_args()

def check_file(path: Path):
    if not path.is_file:
        logger.error(f"File {path} is not exist!")
        exit(-1)


def create_hash(in_dir_path: Path):
    core = Core()
    models = in_dir_path.rglob("*.xml")
    for model_path in models:
        bin_path = model_path.with_suffix(BIN_EXTENSION)
        meta_path = model_path.with_suffix(META_EXTENSION)

        check_file(model_path)
        check_file(bin_path)
        check_file(meta_path)

        str_to_hash = str()
        model = core.read_model(model_path)
        for input in model.inputs:
            str_to_hash += str(len(input.partial_shape)) + str(input.element_type) + str(input.node.type_info) + str(input.partial_shape.is_dynamic)
        for node in model.get_ordered_ops():
            str_to_hash += str(node.type_info)
        for output in model.outputs:      
            str_to_hash += str(len(output.partial_shape)) + str(output.element_type) + str(output.node.type_info) + str(input.partial_shape.is_dynamic)
        
        ports_info = ET.parse(meta_path).getroot().find("ports_info")
        str_to_hash += ET.tostring(ports_info).decode('utf8');        

        old_name = model_path
        new_name = model_path.name[:model_path.name.find('_') + 1] + str(sha256(str_to_hash.encode('utf-8')).hexdigest())

        model_path.rename(Path(model_path.parent, new_name + XML_EXTENSION))
        meta_path.rename(Path(meta_path.parent, new_name + META_EXTENSION))
        bin_path.rename(Path(bin_path.parent, new_name + BIN_EXTENSION))

        logger.info(f"{old_name} -> {new_name}")

if __name__=="__main__":
    args = parse_arguments()
    for in_dir in args.input_dir:
        if not Path(in_dir).is_dir:
            logger.error(f"Directory {in_dir} is not exist!")
            exit(-1)
        logger.info(f"Starting to rename models in {in_dir}")
        create_hash(Path(in_dir))
    logger.info("The run is successfully completed")


