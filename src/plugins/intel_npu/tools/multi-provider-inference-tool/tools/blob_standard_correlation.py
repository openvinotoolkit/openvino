#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import argparse
import json
import os
import sys

from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import bin_diff
from params import TensorsInfoPrinter

def get_model_name(model_path):
    return os.path.basename(model_path).split('.')[0]

class RuntimeErrorExt(RuntimeError):
    def __init__(self, error_code, message):
        super().__init__(message)
        self.error_code = error_code

def get_blobs_std_correlation(left_provider, right_provider, model_path, case_num):
    left_base_dir = Path(left_provider, str(case_num))
    right_base_dir = Path(right_provider, str(case_num))

    printer = TensorsInfoPrinter()
    model_name = get_model_name(model_path)

    # Do not throw exception, error code and description are only expected
    result = {"error_code" : 0, "error_description" : "success"}
    try:
        lhs_model_tensors_descr = printer.deserialize_output_tensor_descriptions(left_base_dir, model_name)
        rhs_model_tensors_descr = printer.deserialize_output_tensor_descriptions(right_base_dir, model_name)

        if len(lhs_model_tensors_descr) != len(rhs_model_tensors_descr):
            raise RuntimeErrorExt(-2, f"Deserialized tensors info of providers: \"{left_provider}\" and \"{right_provider}\" do not match as they contain different amount of model outputs: {len(lhs_model_tensors_descr.keys())} and {len(rhs_model_tensors_descr.keys())} respectively")

        if lhs_model_tensors_descr.keys() != rhs_model_tensors_descr.keys():
            raise RuntimeErrorExt(-2, f"Deserialized tensors info of providers: \"{left_provider}\" and \"{right_provider}\" do not match as they contain different names of model outputs: {lhs_model_tensors_descr.keys()} and {rhs_model_tensors_descr.keys()} respectively")

        for output_name in lhs_model_tensors_descr.keys():
            if "files" not in lhs_model_tensors_descr[output_name].keys() or "files" not in rhs_model_tensors_descr[output_name].keys():
                raise RuntimeErrorExt(-2, f"Deserialized tensors info of providers: \"{left_provider}\" and \"{right_provider}\" do not contain the important attribute \"files\", available attributes for providers: {lhs_model_tensors_descr.keys()} and {rhs_model_tensors_descr.keys()} respectively")

        result["lhs_files"] = []
        result["rhs_files"] = []
        result["std_correlation"] = []
        for lhs_output, rhs_output in zip(lhs_model_tensors_descr.keys(), rhs_model_tensors_descr.keys()):
            assert lhs_output == rhs_output, f"Deserialized tensors info of providers: \"{left_provider}\" and \"{right_provider}\" contain different model outputs: {lhs_model_tensors_descr.keys()} and {rhs_model_tensors_descr.keys()} respectively"

            lhs_data = lhs_model_tensors_descr[lhs_output]
            rhs_data = rhs_model_tensors_descr[rhs_output]

            if lhs_data["shape"] != rhs_data["shape"]:
                raise RuntimeErrorExt(-2, f"Deserialized tensors info of providers: \"{left_provider}\" and \"{right_provider}\" is incomparable, as outputs: \"{lhs_output}\" have different shapes: {lhs_data} and {rhs_data}")
            if lhs_data["element_type"] != rhs_data["element_type"]:
                raise RuntimeErrorExt(-2, f"Deserialized tensors info of providers: \"{left_provider}\" and \"{right_provider}\" is incomparable, as outputs: \"{lhs_output}\" have different element_types: {lhs_data} and {rhs_data}")
            if len(lhs_data["files"]) != len(rhs_data["files"]):
                raise RuntimeErrorExt(-2, f"Deserialized tensors info of providers: \"{left_provider}\" and \"{right_provider}\" is incomparable, as outputs: \"{lhs_output}\" have different amount of files: {lhs_data} and {rhs_data}")
            for lhs_blob, rhs_blob in zip(lhs_data["files"], rhs_data["files"]):
                # Writing files names before doing the check and failing after, gives us possibility
                # to report which files we are expecting to exist, so that increases troubleshooting-abilitiy
                # Thus if someone see files and no correlation for them and the error_code is not 0,
                # it means that we have had an error processing this pair of files from arrays, which haven't been found.
                # So that, we could differenciate which output has been failed exactly, providing a model has multiple outputs
                result["lhs_files"].append(lhs_blob)
                result["rhs_files"].append(rhs_blob)
                result["std_correlation"].append("NaN")
                try:
                    result["std_correlation"][-1] = (bin_diff.compare_blobs(lhs_blob, rhs_blob))
                except Exception as ex:
                    raise RuntimeErrorExt(-3, str(ex))
    except RuntimeErrorExt as ex:
        result["error_description"] = str(ex)
        result["error_code"] = ex.error_code
    except RuntimeError as ex:
        result["error_description"] = str(ex)
        result["error_code"] = -1
    finally:
        pass
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "left_provider",
        help="An inference provider name ",
        default=None
    )
    parser.add_argument(
        "right_provider",
        help="An inference provider name",
        default=None
    )

    parser.add_argument("model", help="A model name or a path to a model contained file", default=None)
    args = parser.parse_args()

    case_num = int(0)
    results = get_blobs_std_correlation(args.left_provider, args.right_provider, args.model, case_num)
    print(json.dumps(results))
