#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import argparse
import json
import os
import sys
import textwrap

import mpit_providers
from params import UseCaseFiles
from params import Config
from params import FilesStorage
from params import ModelInfo
from params import ModelInfoPrinter
from params import if_file
from params import TensorsInfoPrinter

from __version__ import __version__

def parse_inputs(inputs_str):
    return inputs_str.strip().split(";")

try:
    mpit_providers.initialize()
except Exception as ex:
    print(f"ERROR: The application is inoperable, error: {ex}", file=sys.stderr)
    exit(-1)

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-m", "--model", help="Path to model file (.xml)")
parser.add_argument(
    "-p",
    "--provider",
    help="An inference provider, available:\n\t" +'\n\t'.join(mpit_providers.get_avaialable_providers()),
    default=None,
)
parser.add_argument("-i", "--inputs", help=FilesStorage.source_description)
parser.add_argument("-ppm", "--preprocess_model", help=ModelInfo.model_description)
parser.add_argument(
    "-c",
    "--provider_config",
    help=Config.config_description,
)
parser.add_argument('-v', '--version', action='version', version='%(prog)s_' + __version__)

args = parser.parse_args()

if not args.model:
    print("ERROR: Path to model file is missing!", file=sys.stderr)
    parser.print_help()
    exit(1)

if args.provider is None:
    print("ERROR: Inference providers are empty!", file=sys.stderr)
    parser.print_help()
    exit(1)

# check if model exists
if not if_file(args.model):
    print(f"ERROR: Missing model file by path: {args.model}", file=sys.stderr)
    exit(1)

preprocess_model_request_data = ModelInfo(args.preprocess_model)
provider_model_config = Config(args.provider_config)

ctx = mpit_providers.create_provider_ctx(args.provider)
provider = mpit_providers.create_provider_for_model(ctx, args.model)
model = mpit_providers.create_model(provider, preprocess_model_request_data, provider_model_config)
model_info = model.get_model_info()

model_printer = ModelInfoPrinter()
printable_data = model_printer.serialize_model_info(args.provider, args.model, model_info)

# print only model info when no input files are specified
if args.inputs is None:
    print(f"{printable_data}")
    exit(0)

print (f"Model I/O nodes info:")
print(f"{printable_data}")

files_per_uc = UseCaseFiles()
files_per_uc.parse_inputs(args.inputs)


use_case_num = 0
input_files = {}
for files_per_model in files_per_uc.files_per_case:
    input_files[use_case_num] = files_per_model.inputs()
    use_case_num += 1

input_tensors_per_case = {}
tensor_info = {}
for case_num in range(0, max(use_case_num, 1)):
    # at least one usecase with randomly generated input if input is absent
    input_tensors_per_case[case_num] = model.prepare_input_tensors(input_files[case_num])

    # collect tensor + model info for serialization
    tensor_info[case_num] = []
    for input_name, tensor in input_tensors_per_case[case_num].items():
        tensor_info_from_provider = provider.get_tensor_info(tensor)
        tensor_info_from_provider.set_type("input")
        tensor_input_info = dict(model_info.get_model_io_info(input_name))
        tensor_input_info.update(tensor_info_from_provider.info)
        tensor_input_info["source"] = input_name
        tensor_input_info["input_files"] = input_files[case_num][input_name]
        tensor_info[case_num].append(tensor_input_info)

printer = TensorsInfoPrinter()

# print inputs info
print (f"Input tensor infos:")
for case_num in range(0, max(use_case_num, 1)):
    print(json.dumps(printer.get_printable_input_tensor_info(tensor_info[case_num]),indent=4))
print("")

# serialize input tensors into files
for case_num in range(0, max(use_case_num, 1)):
    base_dir = os.path.join(args.provider,str(case_num))
    serialized_file_paths, input_info_path, input_info_dump_path = printer.serialize_tensors_by_type(base_dir, tensor_info[case_num], "input")
    print(f"Use Case[{case_num}]:")
    print("input tensors serialized by paths: ")
    for fp in serialized_file_paths:
        print(f"\t{fp}")
    print("input JSON descriptions(`-i` param compatible): ")
    print(f"\t{input_info_path}")
    print(f"\t{input_info_dump_path}")

# start inference
output_tensors_per_case = {}
for case_num in range(0, max(use_case_num, 1)):
    output_tensors_per_case[case_num] = mpit_providers.infer(model, input_tensors_per_case[case_num])

# collect output tensors info for serializarion
output_tensor_info = {}
for case_num in range(0, max(use_case_num, 1)):
    output_tensor_info[case_num] = []
    for output_name, tensor in output_tensors_per_case[case_num].items():
        tensor_info_from_provider = provider.get_tensor_info(tensor)
        tensor_info_from_provider.set_type("output")
        if output_name in model_info.get_model_io_names():
            tensor_input_info = dict(model_info.get_model_io_info(output_name))
            tensor_input_info.update(tensor_info_from_provider.info)
        else:
            tensor_input_info = tensor_info_from_provider.info
        tensor_input_info["source"] = output_name
        output_tensor_info[case_num].append(tensor_input_info)


# serialize output tensors
for case_num in range(0, max(use_case_num, 1)):
    base_dir = os.path.join(args.provider,str(case_num))
    serialzied_output_tensors = printer.serialize_tensors_by_type(base_dir, output_tensor_info[case_num], "output")

    print(f"Use Case[{case_num}]:")
    print("reference tensors serialized by paths: ")
    for fp in serialzied_output_tensors:
        print(f"\t{fp}")
