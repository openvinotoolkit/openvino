# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import defusedxml.ElementTree as ET

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
from utils.conformance_utils import get_logger, set_env_variable
from utils.constants import PY_OPENVINO, LD_LIB_PATH_NAME, PYTHON_NAME, REL_WEIGHTS_FILENAME, REL_WEIGHTS_REPLACE_STR
from utils.file_utils import get_ov_path, find_latest_dir
import defusedxml.ElementTree as ET

import os
import re
import errno

logger = get_logger('rename_conformance_ir')

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
        logger.warning("Set the following env varibles to rename conformance ir based on hash: ")
        logger.warning(f'PYTHONPATH={env["PYTHONPATH"]}')
        logger.warning(f'{LD_LIB_PATH_NAME}={env[LD_LIB_PATH_NAME]}')
        exit(0)
    else:
        print(f'Impossible to run the tool! PyOpenVINO was not built!')
        exit(-1)
    

XML_EXTENSION = ".xml"
BIN_EXTENSION = ".bin"
META_EXTENSION = ".meta"

@dataclass
class TestStructure:
    dynamic: float = 0.0
    static: float = 0.0

def parse_arguments():
    parser = ArgumentParser()

    in_dir_help = "Path/s to input directory"
    rel_weights_dir = "Path to dir to save rel_weights_file"

    parser.add_argument("--input_dir", help=in_dir_help, nargs="*", required=True)
    parser.add_argument("--rel_weights_dir", help=rel_weights_dir, type=str, default=None, required=False)

    return parser.parse_args()

def check_file(path: Path):
    if not path.is_file:
        logger.error(f"File {path} is not exist or can't be opened!")
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), path)

def generate_op_name(type_info):
    op_name = type_info.name
    op_version = type_info.version_id.replace('opset', '')
    return f"{op_name}-{op_version}"

def get_rel_weight(meta_info_file:Path):
    try:
        meta_info_root = ET.parse(meta_info_file).getroot()
        graph_priority_node = meta_info_root.find("graph_priority")
        value_attrib = float(graph_priority_node.attrib.get("value"))
        return value_attrib
    except:
        logger.error(f"Meta info {meta_info_file} is incorrect!")
        return 1

def update_rel_weight(meta_info_file:Path, additional_value: float):
    try:
        meta_info_root = ET.parse(meta_info_file).getroot()
        graph_priority_node = meta_info_root.find("graph_priority")
        value_attrib = float(graph_priority_node.attrib.get("value"))
        graph_priority_node.set("value", str(value_attrib + additional_value))
        with open(meta_info_file, "w") as xml_file:
            xml_file.write(ET.tostring(meta_info_root).decode('utf8'))
        # logger.info(f"Meta info file {meta_info_file} was updated")
    except:
        logger.error(f"Meta info {meta_info_file} is incorrect!")

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
        meta_path = model_path.with_suffix(META_EXTENSION)

        try:
            check_file(model_path)
            check_file(bin_path)
            check_file(meta_path)

            str_to_hash = str()
            rel_weight = get_rel_weight(meta_path)

            try:
                model = core.read_model(model_path)
                for node in model.get_ordered_ops():
                    op_name = generate_op_name(node.get_type_info())
                    if is_report_op(op_name):
                        if not op_name in operations.keys():
                            operations.update({op_name: TestStructure()})
                        # add op/subgraphs, dynamic/static and extractor_name to hash
                        model_dir, _ = os.path.split(model_path)
                        model_dir = str(model_dir).replace(model_prefix, "")
                        if op_name in model_dir:
                            model_dir = model_dir[:model_dir.find(op_name):]
                        model_dir = model_dir[:-1:]
                        model_dir = model_dir.replace(os.path.sep, "_")
                        str_to_hash += model_dir
                        # upgrade expected rel passrates files
                        if "static" in str(model_path):
                            operations[op_name].static += rel_weight
                        elif "dynamic" in str(model_path):
                            operations[op_name].dynamic += rel_weight
                    str_to_hash += generate_node_hash(node)
                    try:
                        for body_node in node.get_function().get_ordered_ops():
                            str_to_hash += generate_node_hash(body_node)
                    except:
                        pass
            except:
                logger.error(f"Impossible to create hash for {model_path}")

            try:
                # check only parameters/constant structures
                for input in ET.parse(meta_path).getroot().find("input_info"):
                    for attrib in input.attrib:
                        if attrib == "convert_to_const":
                            str_to_hash += input.attrib.get(attrib)
            except:
                logger.error(f"Impossible to add input_info to hash for {model_path}")

            old_name = model_path
            new_name = str(sha256(str_to_hash.encode('utf-8')).hexdigest())

            new_meta_path = Path(meta_path.parent, new_name + META_EXTENSION)
            new_xml_path = Path(model_path.parent, new_name + XML_EXTENSION)
            new_bin_path = Path(bin_path.parent, new_name + BIN_EXTENSION)

            if not os.path.isfile(new_meta_path):
                model_path.rename(new_xml_path)
                meta_path.rename(new_meta_path)
                bin_path.rename(new_bin_path)
                # TODO: if some models are still not renaming, create new file and remove old file
                # logger.info(f"{old_name} -> {new_name}")
            elif old_name != new_xml_path:
                # TODO: if some models are still not renaming and there are duplicates, remove files here
                logger.warning(f"Could not rename model {old_name} ! Model file name already exists {new_xml_path} ")
                update_rel_weight(new_meta_path, rel_weight)
        except:
            pass
    return operations

def save_rel_weights(rel_weights_dir:Path, operations: dict):
    if not rel_weights_dir.is_dir:
        logger.info(f"Create rel weight_dir: {rel_weights_dir}")
        os.mkdir(rel_weights_dir)
    rel_weights_path = os.path.join(rel_weights_dir, REL_WEIGHTS_FILENAME.replace(REL_WEIGHTS_REPLACE_STR, ""))
    dyn_rel_weights_path = os.path.join(rel_weights_dir, REL_WEIGHTS_FILENAME.replace(REL_WEIGHTS_REPLACE_STR, "dynamic"))
    static_rel_weights_path = os.path.join(rel_weights_dir, REL_WEIGHTS_FILENAME.replace(REL_WEIGHTS_REPLACE_STR, "static"))

    rel_weights_file = open(rel_weights_path, "w")
    dyn_rel_weights_file = open(dyn_rel_weights_path, "w")
    static_rel_weights_file = open(static_rel_weights_path, "w")

    for op, rel_weight in operations.items():
        if rel_weight.dynamic != 0:
            dyn_rel_weights_file.write(f"{op}:{rel_weight.dynamic}\n")
        if rel_weight.static != 0:
            static_rel_weights_file.write(f"{op}:{rel_weight.static}\n")
        rel_weights_file.write((f"{op}:{rel_weight.static + rel_weight.dynamic}\n"))
    
    rel_weights_file.close()
    dyn_rel_weights_file.close()
    static_rel_weights_file.close()

    logger.info(f"Relative weights are saved to {rel_weights_path}, {dyn_rel_weights_path}, {static_rel_weights_path}")
    return rel_weights_path, dyn_rel_weights_path, static_rel_weights_path

if __name__=="__main__":
    args = parse_arguments()
    operations = dict()
    rel_weights_dir = None
    if not args.rel_weights_dir is None:
        rel_weights_dir = Path(args.rel_weights_dir)
        if not rel_weights_dir.is_dir():
            logger.info(f"Create rel weight_dir: {rel_weights_dir}")
            os.mkdir(rel_weights_dir)

    for in_dir in args.input_dir:
        if not Path(in_dir).is_dir():
            logger.error(f"Directory {in_dir} is not exist!")
            continue
        # logger.info(f"Starting to rename models in {in_dir}")
        operations = create_hash(Path(in_dir), operations)
    
    if not rel_weights_dir is None:
        save_rel_weights(rel_weights_dir, operations)

    logger.info("The run is successfully completed")