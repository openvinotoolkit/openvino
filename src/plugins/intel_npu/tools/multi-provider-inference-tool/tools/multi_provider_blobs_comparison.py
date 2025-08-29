#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import argparse
import json
import os
import pathlib
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import bin_diff
from collections import defaultdict
from params import TensorsInfoPrinter

def get_model_name(model_path):
    return os.path.basename(model_path).split('.')[0]

class RuntimeErrorExt(RuntimeError):
    def __init__(self, error_code, message):
        super().__init__(message)
        self.error_code = error_code

def initialize_provider_output_result_data(deserialied_output_data, leaf_default_values):
    output_result = {"files": [],
                     "shape": deserialied_output_data["shape"],
                     "element_type": deserialied_output_data["element_type"],
                     "data": {},
                     "not_found_data": defaultdict(dict),
                     "status": []}
    for blob_file_path in deserialied_output_data["files"]:
        if not os.path.isfile(blob_file_path):
            output_result["status"].append(f"File not found: {blob_file_path}")
            continue

        std_correlation_default_value = "NaN"
        if "std_correlation" in leaf_default_values.keys():
            std_correlation_default_value = leaf_default_values["std_correlation"]

        blob_file_name = os.path.basename(blob_file_path)
        output_result["data"][blob_file_name] = {"path": blob_file_path, "std_correlation": std_correlation_default_value}
        output_result["files"].append(blob_file_name)
    return output_result

def initialize_provider_result_data(provider_name, deserialized_model_tensors_descr, leaf_default_values):
    provider_result = {"outputs" : [], "data" : {}, "not_found_data": defaultdict(dict), "status" : []}

    for output_name, output_data in deserialized_model_tensors_descr.items():
        # sanity check
        prev_error_num = len(provider_result["status"])
        for field_to_check in ["shape","element_type", "files"]:
            if field_to_check not in output_data.keys():
                provider_result["status"].append(f"Deserialized tensors info of provider: \"{provider_name}\" doesn't contain important field \"{field_to_check}\": {output_data}. Result is malformed")
        if prev_error_num != len(provider_result["status"]):
            # nothing to add: output results are malformed
            continue

        # initialize data
        provider_result["data"][output_name] = initialize_provider_output_result_data(output_data, leaf_default_values)
        provider_result["outputs"].append(output_name)
    return provider_result

def get_comparison_profiver_output_files_result_data(ref_provider_output_blob_file_data, provider_to_cmp_output_blob_file_data):
    provider_to_cmp_output_blob_file_data["std_correlation"] = bin_diff.compare_blobs(ref_provider_output_blob_file_data["path"], provider_to_cmp_output_blob_file_data["path"])
    return provider_to_cmp_output_blob_file_data

def get_comparison_provider_output_result_data(ref_provider_name, ref_provider_output_data, provider_to_cmp_name, provider_to_cmp_output_data):
    for field_to_compare in ["shape", "element_type"]:
        if ref_provider_output_data[field_to_compare] != provider_to_cmp_output_data[field_to_compare]:
            provider_to_cmp_output_data["status"].append(f"Incorrect \"{field_to_compare}\": {provider_to_cmp_output_data[field_to_compare]}, expected: {ref_provider_output_data[field_to_compare]}")
            ref_provider_output_data["status"].append(f"Incorrect \"{field_to_compare}\": {ref_provider_output_data[field_to_compare]}, expected: {provider_to_cmp_output_data[field_to_compare]}")
    # If a reference output blob file is found in rhs provider, we will conduct compasion of these files.
    # Otherwise, we do nothing.

    # Eventually, if rhs provider has different outputs blob files than the reference provider,
    # then the data cannot be compared to the reference data and it will holds NaN,
    # which is respobisility of a consumer to decide how they will evaluate that

    # Ultimately, we compare only outputs/files that are coherent and skip unmatching outputs/files
    for f in ref_provider_output_data["files"]:
        if f in provider_to_cmp_output_data["data"].keys():
            try:
                provider_to_cmp_output_data["data"][f] = get_comparison_profiver_output_files_result_data(ref_provider_output_data["data"][f], provider_to_cmp_output_data["data"][f])
            except RuntimeError as ex:
                provider_to_cmp_output_data["status"].append(f"Cannot compare the file: {f}, err: {ex}")
        else:
            if "provider" not in provider_to_cmp_output_data["not_found_data"][f].keys():
                provider_to_cmp_output_data["not_found_data"][f]["provider"] = []
            provider_to_cmp_output_data["not_found_data"][f]["provider"].append(ref_provider_name)

    for f in provider_to_cmp_output_data["files"]:
        if f not in ref_provider_output_data["data"].keys():
            if "provider" not in ref_provider_output_data["not_found_data"][f].keys():
                ref_provider_output_data["not_found_data"][f]["provider"] = []
            ref_provider_output_data["not_found_data"][f]["provider"].append(provider_to_cmp_name)
    return provider_to_cmp_output_data

def get_comparison_provider_result_data(ref_provider_name, ref_provider_result_data, provider_to_cmp_name, provider_to_cmp_result_data):
    for ref_output_name in ref_provider_result_data["outputs"]:
        # If a reference output is found in rhs provider, we will conduct compasion of blobs.
        # Otherwise, we do nothing.

        # Eventually, if rhs provider has different outputs than the reference provider,
        # then the data cannot be compared to the reference data and it will holds NaN,
        # which is respobisility of a consumer to decide how they will evaluate that

        # Ultimately, we compare only outputs/files that are coherent and skip unmatching outputs/files
        if ref_output_name in provider_to_cmp_result_data["data"].keys():
            provider_to_cmp_result_data["data"][ref_output_name] = get_comparison_provider_output_result_data(ref_provider_name, ref_provider_result_data["data"][ref_output_name], provider_to_cmp_name, provider_to_cmp_result_data["data"][ref_output_name])
        else:
            if "provider" not in provider_to_cmp_result_data["not_found_data"][ref_output_name].keys():
                provider_to_cmp_result_data["not_found_data"][ref_output_name]["provider"] = []
            provider_to_cmp_result_data["not_found_data"][ref_output_name]["provider"].append(ref_provider_name)

    # Fill outputs which are missed in reference data but exist in provider_to_cmp
    for output_to_cmp_name in provider_to_cmp_result_data["outputs"]:
        if output_to_cmp_name not in ref_provider_result_data["data"].keys():
            if "provider" not in ref_provider_result_data["not_found_data"][output_to_cmp_name].keys():
                ref_provider_result_data["not_found_data"][output_to_cmp_name]["provider"] = []
            ref_provider_result_data["not_found_data"][output_to_cmp_name]["provider"].append(provider_to_cmp_name)
    return provider_to_cmp_result_data

def multi_provider_result_comparator(ref_provider, providers_to_compare, model_path, case_num):
    printer = TensorsInfoPrinter()
    model_name = get_model_name(model_path)
    result = {"providers" : [], "data" : {}, "status": [], "not_found_data":{}}

    # gather result for a reference provider
    ref_provider_base_dir = os.path.join(ref_provider, str(case_num))
    try:
        # read data from provider affiliated directories
        ref_provider_model_tensors_descr = printer.deserialize_output_tensor_descriptions(ref_provider_base_dir, model_name)

        # fill LHS default values
        result["data"][ref_provider] = initialize_provider_result_data(ref_provider, ref_provider_model_tensors_descr, {"std_correlation": 1})
        result["providers"].append(ref_provider)
    except RuntimeError as ex:
        result["status"].append(f"Cannot find results for the reference provider: {ref_provider}, err: {ex}")
        result["not_found_data"][ref_provider] = {}

    # gather results for providers to compare
    for p in providers_to_compare:
        provider_base_dir = os.path.join(p, str(case_num))
        try:
            # read data from provider affiliated directories
            provider_model_tensors_descr = printer.deserialize_output_tensor_descriptions(provider_base_dir, model_name)

            # fill default values
            result["data"][p] = initialize_provider_result_data(p, provider_model_tensors_descr, {"std_correlation": "NaN"})
            result["providers"].append(p)
        except Exception as ex:
            result["status"].append(f"Cannot find results for provider: {p}, err: {str(ex)}")
            result["not_found_data"][p] = {}

    # make comparison between reference and right-hand-side providers
    if len(result["providers"]) == 0:
        return result

    if result["providers"][0] == ref_provider and len(result["providers"]) > 1:
        for p in result["providers"][1:]:
            result["data"][p] = get_comparison_provider_result_data(ref_provider, result["data"][ref_provider], p, result["data"][p])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "ref_provider",
        help="The name of a referecnce provider",
        default=None
    )
    parser.add_argument(
        "providers_to_compare",
        nargs='+',
        help="An inference provider names to compare to the reference provider",
        default=None
    )

    parser.add_argument("model", help="A model name or a path to a model contained file", default=None)
    args = parser.parse_args()

    case_num = int(0)
    results = multi_provider_result_comparator(args.ref_provider, args.providers_to_compare, args.model, case_num)
    print(json.dumps(results, indent=4))
