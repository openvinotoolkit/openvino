# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
from utils.conformance_utils import get_logger, set_env_variable
from utils.constants import PY_OPENVINO, LD_LIB_PATH_NAME
from utils.file_utils import get_ov_path

import os
import re
import errno

logger = get_logger('Calculate_IR_hash')

try:
    from openvino.runtime import Core
except:
    script_dir, _ = os.path.split(os.path.abspath(__file__))
    ov_bin_path = get_ov_path(script_dir, None, True)
    if PY_OPENVINO in os.listdir(ov_bin_path):
        env = os.environ
        py_ov = os.path.join(ov_bin_path, PY_OPENVINO)

        env = set_env_variable(env, "PYTHONPATH", py_ov)
        env = set_env_variable(env, LD_LIB_PATH_NAME, ov_bin_path)
        logger.warning("Set the following env varibles to calculate the IR-based hash: ")
        logger.warning(f'PYTHONPATH={env["PYTHONPATH"]}')
        logger.warning(f'{LD_LIB_PATH_NAME}={env[LD_LIB_PATH_NAME]}')
        exit(0)
    else:
        logger.error(f'Impossible to run the tool! PyOpenVINO was not found!')
        exit(-1)

XML_EXTENSION = ".xml"
BIN_EXTENSION = ".bin"

@dataclass
class TestStructure:
    dynamic: float = 0.0
    static: float = 0.0

def parse_arguments():
    parser = ArgumentParser()

    in_dir_help = "Path/s to the input directory"

    parser.add_argument("--input_dir", help=in_dir_help, nargs="*", required=True)

    return parser.parse_args()

def check_file(path: Path):
    if not path.is_file:
        logger.error(f"File {path} does not exist or can't be opened!")
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), path)

def generate_op_name(type_info):
    op_name = type_info.name
    op_version = type_info.version_id.replace('opset', '')
    return f"{op_name}-{op_version}"

def is_report_op(op_name:str):
    if "Parameter-1" == op_name or "Result-1" == op_name or "Constant-1" == op_name:
        return False
    return True

def generate_node_hash(node):
    str_to_hash = ""
    for input in node.inputs():
        input_node = input.get_node()

        shape_str = ""
        try:
            str_to_hash += re.sub(r"[\s+\[\]\{\}\']", "", str(node.get_attributes()))
        except:
            logger.error(f"Impossible to get attributes for {node.name}")
        try:
            partial_shape = input.get_partial_shape()

            if 'Convolution' in str(input_node.get_type_info().name):
                offset = 2
                if 'GroupConvolution' in str(input_node.get_type_info().name) or\
                   'GroupConvolutionBackpropData' in str(input_node.get_type_info().name):
                    offset = 3
                shape_str += '[' + ','.join([str(val) for val in list(partial_shape)[offset:]]) + ']'

            shape_str += str(len(partial_shape))
            shape_str += str(partial_shape.is_dynamic)
        except:
            logger.error(f"Impossible to get input_shape for {input_node.name}")

        str_to_hash += shape_str + str(input.get_element_type().get_type_name()) + \
            str(input_node.get_type_info().name) + str(input_node.get_type_info().version_id)

    for output in node.outputs():
        output_node = output.get_node()

        shape_str = ""
        try:
            partial_shape = output.get_partial_shape()
            shape_str += str(len(partial_shape))
            shape_str += str(partial_shape.is_dynamic)
        except:
            logger.error(f"Impossible to get output_shape for {output.names.pop()}")

        str_to_hash += shape_str + str(output.get_element_type().get_type_name()) + \
            str(output_node.get_type_info().name) + str(output_node.get_type_info().version_id)

    return str_to_hash

def create_hash(in_dir_path: Path, operations=dict()):
    core = Core()
    models = in_dir_path.rglob("*.xml")
    models = sorted(models)
    model_prefix = os.path.commonprefix(models)
    for model_path in models:
        bin_path = model_path.with_suffix(BIN_EXTENSION)

        try:
            check_file(model_path)
            check_file(bin_path)

            str_to_hash = str()

            try:
                model = core.read_model(model_path)
                for node in model.get_ordered_ops():
                    op_name = generate_op_name(node.get_type_info())
                    if is_report_op(op_name):
                        if not op_name in operations.keys():
                            operations.update({op_name: TestStructure()})
                        # add op/subgraphs and extractor_name to hash
                        model_dir, _ = os.path.split(model_path)
                        model_dir = str(model_dir).replace(model_prefix, "")
                        if op_name in model_dir:
                            model_dir = model_dir[:model_dir.find(op_name):]
                        model_dir = model_dir[:-1:]
                        model_dir = model_dir.replace(os.path.sep, "_")
                        str_to_hash += model_dir

                    str_to_hash += generate_node_hash(node)

                    try:
                        for body_node in node.get_function().get_ordered_ops():
                            str_to_hash += generate_node_hash(body_node)
                    except:
                        pass
            except:
                logger.error(f"Impossible to create hash for the{model_path}")

            if not str_to_hash:
                hash = "INVALID"
            else:
                hash = str(sha256(str_to_hash.encode('utf-8')).hexdigest())

            logger.info(f"Generated hash for the {model_path}: {hash}")
        except:
            pass
    return operations

if __name__=="__main__":
    args = parse_arguments()
    operations = dict()
    rel_weights_dir = None

    for in_dir in args.input_dir:
        if not Path(in_dir).is_dir():
            logger.error(f"Specified directory {in_dir} does not exist!")
            continue
        operations = create_hash(Path(in_dir), operations)

    logger.info(f"The run was completed!")
