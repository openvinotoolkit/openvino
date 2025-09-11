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

from pathlib import Path

import mpit_providers
from params import UseCaseFiles
from params import Config
from params import FilesStorage
from params import ModelInfo
from params import ModelInfoPrinter
from params import if_file
from params import TensorsInfoPrinter

from __version__ import __version__

try:
    mpit_providers.initialize()
except Exception as ex:
    print(f"ERROR: The application is inoperable, error: {ex}", file=sys.stderr)
    exit(-1)

def get_valid_command_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="Path to a model file", type=Path)
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
    cmd_args = parser.parse_args()

    if not cmd_args.model:
        print("ERROR: Path to model file is missing!", file=sys.stderr)
        parser.print_help()
        exit(1)

    if not cmd_args.model.is_file():
        print(f"ERROR: Missing a model file by the path: {cmd_args.model}", file=sys.stderr)
        exit(1)

    if cmd_args.provider is None:
        print("ERROR: Inference providers are empty!", file=sys.stderr)
        parser.print_help()
        exit(1)
    return cmd_args


def get_usecase_files(input_files_cmd_line):
    files_per_uc = UseCaseFiles(input_files_cmd_line)

    usecase_num = 0
    input_files = {}
    for files_per_model in files_per_uc.files_per_case:
        input_files[usecase_num] = files_per_model.inputs()
        usecase_num += 1
    return input_files

def get_input_tensor_metadata(provider, model_info, input_tensors_per_case):
    tensor_info = {}
    for case_num in range(0, len(input_tensors_per_case)):
        tensor_info[case_num] = []
        for input_name, tensor in input_tensors_per_case[case_num].items():
            tensor_info_from_provider = provider.get_tensor_info(tensor)
            tensor_info_from_provider.set_type("input")
            tensor_input_info = dict(model_info.get_model_io_info(input_name))
            tensor_input_info.update(tensor_info_from_provider.info)
            tensor_input_info["source"] = input_name
            tensor_input_info["input_files"] = input_files[case_num][input_name]
            tensor_info[case_num].append(tensor_input_info)
    return tensor_info

def serialize_inference_input_artefacts(serializer, provider_name, input_tensors_info):
    for case_num in range(0, max(len(input_tensors_info), 1)):
        root_dir = Path(provider_name) / str(case_num)
        serialized_file_paths, input_info_path, input_info_dump_path = serializer.serialize_tensors_by_type(root_dir, input_tensors_info[case_num], "input")
        print(f"Use Case[{case_num}]:")
        print("input tensors serialized by paths: ")
        for fp in serialized_file_paths:
            print(f"\t{fp}")
        print("input JSON descriptions(`-i` param compatible): ")
        print(f"\t{input_info_path}")
        print(f"\t{input_info_dump_path}")

def get_output_tensor_metadata(provider, model_info, output_tensors_per_case):
    output_tensor_info = {}
    for case_num in range(0, len(output_tensors_per_case)):
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
    return output_tensor_info

def serialize_inference_output_artefacts(serializer, provider_name, output_tensors_info):
    for case_num in range(0, max(len(output_tensors_info), 1)):
        root_dir = Path(provider_name) / str(case_num)
        serialzied_output_tensors = serializer.serialize_tensors_by_type(root_dir, output_tensors_info[case_num], "output")

        print(f"Use Case[{case_num}]:")
        print("reference tensors serialized by paths: ")
        for fp in serialzied_output_tensors:
            print(f"\t{fp}")

if __name__ == "__main__":
    args = get_valid_command_arguments()

    # instantiate a specific inference provider and compile a model
    ctx = mpit_providers.create_provider_ctx(args.provider)
    provider = mpit_providers.create_provider_for_model(ctx, args.model)
    model = mpit_providers.create_model(provider, args.provider_config, args.preprocess_model)

    # Expose model inputs/outputs info
    model_info = model.get_model_info()
    model_info_data = ModelInfoPrinter().serialize_model_info(args.provider, args.model, model_info)
    if args.inputs is None:
        # When no inference requested, meaning no input files are specified,
        # print only model info JSON, which can be parsed as a part of command pipelining
        print(f"{model_info_data}")
        exit(0)

    print(f"Model I/O nodes info:")
    print(f"{model_info_data}")

    # convert input files into a tensor format specific for the requested provider
    input_files = get_usecase_files(args.inputs)
    usecase_num = len(input_files)
    input_tensors_per_case = {}
    for case_num in range(0, max(usecase_num, 1)):
        input_tensors_per_case[case_num] = model.prepare_input_tensors(input_files[case_num])

    # Input tensor info being written in a file allows
    # us to load that file as `-i` in subsequent tool invocations to keep
    # persisten inferences history
    tensor_info = get_input_tensor_metadata(provider, model_info, input_tensors_per_case)

    print (f"Input tensor info:")
    printer = TensorsInfoPrinter()
    for case_num in range(0, max(usecase_num, 1)):
        print(json.dumps(printer.get_printable_input_tensor_info(tensor_info[case_num]),indent=4))
    print("")

    # serialize input tensors into files to store the model inference in the history
    serialize_inference_input_artefacts(printer, args.provider, tensor_info)

    # start inference
    output_tensors_per_case = {}
    for case_num in range(0, max(usecase_num, 1)):
        output_tensors_per_case[case_num] = mpit_providers.infer(model, input_tensors_per_case[case_num])


    # Collected output tensors and the model info participate in
    # output tensor meta information creation, which being written in a file allows
    # us to enchance output blobs inter-provider comparison procedure, by
    # providing more precised and self-descriptive comparison results as well
    # as keeping inferences history
    output_tensor_info = get_output_tensor_metadata(provider, model_info, output_tensors_per_case)

    # serialize output tensors into files to store the model inference in the history
    serialize_inference_output_artefacts(printer, args.provider, output_tensor_info)
