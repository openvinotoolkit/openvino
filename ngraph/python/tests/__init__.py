# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import pytest

# test.BACKEND_NAME is a configuration variable determining which
# nGraph backend tests will use. It's set during pytest configuration time.
# See `pytest_configure` hook in `conftest.py` for more details.
BACKEND_NAME = None

# test.ADDITIONAL_MODELS_DIR is a configuration variable providing the path
# with additional ONNX models to load and test import. It's set during pytest
# configuration time. See `pytest_configure` hook in `conftest.py` for more
# details.
ADDITIONAL_MODELS_DIR = None


def xfail_test(reason="Mark the test as expected to fail", strict=True):
    return pytest.mark.xfail(reason=reason, strict=strict)


xfail_issue_34314 = xfail_test(reason="RuntimeError: RNNCell operation has a form that is not "
                               "supported.RNNCell_21204 should be converted to RNNCellIE operation")
skip_segfault = pytest.mark.skip(reason="Segmentation fault error")
xfail_issue_34323 = xfail_test(reason="RuntimeError: data [value] doesn't exist")
xfail_issue_34327 = xfail_test(reason="RuntimeError: '<value>' layer has different "
                                      "IN and OUT channels number")
xfail_issue_35893 = xfail_test(reason="ValueError: could not broadcast input array")
xfail_issue_35911 = xfail_test(reason="Assertion error: Pad model mismatch error")
xfail_issue_35912 = xfail_test(reason="RuntimeError: Error of validate layer: B with type: "
                                      "Pad. Cannot parse parameter pads_end  from IR for layer B. "
                                      "Value -1,0 cannot be casted to int.")
xfail_issue_35914 = xfail_test(reason="IndexError: too many indices for array: "
                                      "array is 0-dimensional, but 1 were indexed")
xfail_issue_35915 = xfail_test(reason="RuntimeError: Eltwise node with unsupported combination "
                                      "of input and output types")
xfail_issue_35916 = xfail_test(reason="RuntimeError: Unsupported input dims count for layer Z")
xfail_issue_35917 = xfail_test(reason="RuntimeError: Unsupported input dims count for "
                                      "layer MatMul")
xfail_issue_35918 = xfail_test(reason="onnx.onnx_cpp2py_export.checker.ValidationError: "
                                      "Mismatched attribute type in 'test_node : alpha'")
xfail_issue_35921 = xfail_test(reason="ValueError - shapes mismatch in gemm")

xfail_issue_35923 = xfail_test(reason="RuntimeError: PReLU without weights is not supported")
xfail_issue_35924 = xfail_test(reason="Assertion error - elu results mismatch")
xfail_issue_35925 = xfail_test(reason="Assertion error - reduction ops results mismatch")
xfail_issue_35926 = xfail_test(reason="RuntimeError: [NOT_IMPLEMENTED] Input image format I64 is "
                                      "not supported yet...")
xfail_issue_35927 = xfail_test(reason="RuntimeError: B has zero dimension that is not allowable")
xfail_issue_35929 = xfail_test(reason="CRuntimeError: Incorrect precision f64!")
xfail_issue_35930 = xfail_test(reason="onnx.onnx_cpp2py_export.checker.ValidationError: "
                                      "Required attribute 'to' is missing.")
xfail_issue_35932 = xfail_test(reason="Assertion error - logsoftmax results mismatch")
xfail_issue_36437 = xfail_test(reason="RuntimeError: Cannot find blob with name: y")
xfail_issue_36476 = xfail_test(reason="RuntimeError: [NOT_IMPLEMENTED] Input image format U32 is "
                               "not supported yet...")
xfail_issue_36478 = xfail_test(reason="RuntimeError: [NOT_IMPLEMENTED] Input image format U64 is "
                               "not supported yet...")
xfail_issue_36479 = xfail_test(reason="Assertion error - basic computation on ndarrays mismatch")
xfail_issue_36480 = xfail_test(reason="RuntimeError: [NOT_FOUND] Unsupported property dummy_option "
                               "by CPU plugin")
xfail_issue_36481 = xfail_test(reason="TypeError: _get_node_factory() takes from 0 to 1 positional "
                               "arguments but 2 were given")
xfail_issue_36483 = xfail_test(reason="RuntimeError: Unsupported primitive of type: "
                               "Ceiling name: Ceiling_22669")
xfail_issue_36485 = xfail_test(reason="RuntimeError: Check 'm_group >= 1' failed at "
                               "/openvino/ngraph/src/ngraph/op/fused/shuffle_channels.cpp:77:")
xfail_issue_36486 = xfail_test(reason="RuntimeError: HardSigmoid operation should be converted "
                                      "to HardSigmoid_IE")
xfail_issue_36487 = xfail_test(reason="Assertion error - mvn operator computation mismatch")


# Model Zoo issues:
xfail_issue_36533 = xfail_test(reason="AssertionError: zoo models results mismatch")
xfail_issue_36534 = xfail_test(reason="RuntimeError: node input index is out of range")
xfail_issue_36535 = xfail_test(reason="RuntimeError: get_shape was called on a descriptor::Tensor "
                                      "with dynamic shape")
xfail_issue_36536 = xfail_test(reason="RuntimeError: can't protect")
xfail_issue_36537 = xfail_test(reason="ngraph.exceptions.UserInputError: (Provided tensor's shape: "
                                      "<value> does not match the expected: <value>")
xfail_issue_36538 = xfail_test(reason="RuntimeError: Check 'PartialShape::broadcast_merge_into( pshape, "
                                      "node->get_input_partial_shape(i), autob)' failed at "
                                      "/openvino/ngraph/src/ngraph/op/util/elementwise_args.cpp:48:")
