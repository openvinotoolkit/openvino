# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from utils.file_utils import prepare_filelist
from utils.constants import XML_EXTENSION, META_EXTENSION
from openvino.runtime import Core
import defusedxml.ElementTree as ET

def parse_arguments():
    parser = ArgumentParser()

    origin_help = "Path to output subgraphs dir"
    ref_help = "Path to refence subgraphs dir"

    parser.add_argument("-o", "--origin", help=origin_help, required=True)
    parser.add_argument("-r", "--reference", help=ref_help, required=True)

    return parser.parse_args()

def list_to_set(in_list: list):
    result = set()
    for list_item in in_list:
        result.add(list_item)
    return result

def models_to_names(files: set):
    core = Core()
    result = dict()
    for model_file in files:
        model = core.read_model(model_file)
        nodes_name = set()
        for node in model.get_ordered_ops():
            if "Result" in node.get_type_info().name or "Parameter" in node.get_type_info().name or "Constant" in node.get_type_info().name:
                continue
            nodes_name.add(node.get_friendly_name())
        result.update({model_file: nodes_name})
    return result

def models_info(files: set):
    result = dict()
    for model_file in files:
        meta_path = model_file.with_suffix(META_EXTENSION)
        meta_info_root = ET.parse(meta_path).getroot()
        model_node = meta_info_root.find("models")
        model_names = set()
        for model_name in model_node:
            model_names.add(model_name.attrib.get("name"))
        result.update({model_file: model_names})
    return result

if __name__ == "__main__":
    args = parse_arguments()
    file_list_orig = prepare_filelist(args.origin, [f"*{XML_EXTENSION}"], False)
    file_list_ref = prepare_filelist(args.reference, [f"*{XML_EXTENSION}"], False)

    orig_irs = list_to_set(file_list_orig)
    ref_irs = list_to_set(file_list_ref)

    orig_dif = orig_irs.difference(ref_irs)
    ref_dif = ref_irs.difference(orig_irs)

    node_names_by_ir_orig = models_to_names(orig_dif)
    node_names_by_ir_ref = models_to_names(ref_dif)

    model_info_orig = models_info(orig_dif)
    model_info_ref = models_info(ref_dif)

    result_dict = dict()
    result_intersection = dict()
    result_orig_dif = dict()
    result_ref_dif = dict()
    for ir_name_orig in model_info_orig.keys():
        models_name_orig = model_info_orig[ir_name_orig]
        for ir_name_ref in model_info_ref.keys():
            models_name_ref = model_info_ref[ir_name_ref]
            if len(models_name_orig.intersection(models_name_ref)) == 0:
                continue
            if node_names_by_ir_orig[ir_name_orig] == node_names_by_ir_ref[ir_name_ref]:
                continue
            if len(node_names_by_ir_orig[ir_name_orig].intersection(node_names_by_ir_ref[ir_name_ref])) == 0:
                continue
            result_dict.update({ir_name_orig: ir_name_ref})
            result_intersection.update({ir_name_orig: node_names_by_ir_orig[ir_name_orig].intersection(node_names_by_ir_ref[ir_name_ref])})
            result_orig_dif.update({ir_name_orig: node_names_by_ir_orig[ir_name_orig].difference(node_names_by_ir_ref[ir_name_ref])})
            result_ref_dif.update({ir_name_ref: node_names_by_ir_ref[ir_name_ref].difference(node_names_by_ir_orig[ir_name_orig])})
    for ir_name_orig in result_dict.keys():
        ir_name_ref = result_dict[ir_name_orig]
        print(f"Intersected IRs: {ir_name_orig} vs {ir_name_ref}")
        print(f"Intersected functional operation names: {result_intersection[ir_name_orig]}")
        print(f"Orig new operation names: {result_orig_dif[ir_name_orig]}")
        print(f"Ref new operation names: {result_ref_dif[ir_name_ref]}")



