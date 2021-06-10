# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

# test.BACKEND_NAME is a configuration variable determining which
# nGraph backend tests will use. It's set during pytest configuration time.
# See `pytest_configure` hook in `conftest.py` for more details.
BACKEND_NAME = None

# test.MODEL_ZOO_DIR is a configuration variable providing the path
# to the ZOO of ONNX models to test. It's set during pytest configuration time.
# See `pytest_configure` hook in `conftest.py` for more
# details.
MODEL_ZOO_DIR = None

# test.MODEL_ZOO_XFAIL is a configuration variable which enable xfails for model zoo.
MODEL_ZOO_XFAIL = False


def xfail_test(reason="Mark the test as expected to fail", strict=True):
    return pytest.mark.xfail(reason=reason, strict=strict)


skip_segfault = pytest.mark.skip(reason="Segmentation fault error")
xfail_issue_33488 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "MaxUnpool")
xfail_issue_33512 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Einsum")
xfail_issue_33535 = xfail_test(reason="nGraph does not support the following ONNX operations:"
                                      "DynamicQuantizeLinear")
xfail_issue_33538 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Scan")
skip_issue_38084 = pytest.mark.skip(reason="Aborted (core dumped) Assertion "
                                           "`(layer->get_output_partial_shape(i).is_static())' failed.")
xfail_issue_33589 = xfail_test(reason="nGraph does not support the following ONNX operations:"
                                      "IsNaN and isInf")
xfail_issue_33595 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Unique")
xfail_issue_33596 = xfail_test(reason="RuntimeError: nGraph does not support different sequence operations:"
                                      "ConcatFromSequence, SequenceConstruct, SequenceAt, SplitToSequence,"
                                      "SequenceEmpty, SequenceInsert, SequenceErase, SequenceLength ")
xfail_issue_33606 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Det")
xfail_issue_33651 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "TfIdfVectorizer")
xfail_issue_33581 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "GatherElements")
xfail_issue_33633 = xfail_test(reason="MaxPool: dilations unsupported")
xfail_issue_35911 = xfail_test(reason="Assertion error: Pad model mismatch error")
xfail_issue_35923 = xfail_test(reason="RuntimeError: PReLU without weights is not supported")
xfail_issue_35927 = xfail_test(reason="RuntimeError: B has zero dimension that is not allowable")
xfail_issue_36486 = xfail_test(reason="RuntimeError: HardSigmoid operation should be converted "
                                      "to HardSigmoid_IE")
xfail_issue_36487 = xfail_test(reason="Assertion error - mvn operator computation mismatch")
xfail_issue_38084 = xfail_test(reason="RuntimeError: AssertionFailed: layer->get_output_partial_shape(i)"
                                      "is_static() nGraph <value> operation with name: <value> cannot be"
                                      "converted to <value> layer with name: <value> because output"
                                      "with index 0 contains dynamic shapes: {<value>}. Try to use "
                                      "CNNNetwork::reshape() method in order to specialize shapes "
                                      "before the conversion.")
xfail_issue_38091 = xfail_test(reason="AssertionError: Mismatched elements")
xfail_issue_38699 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ai.onnx.preview.training.Gradient")
xfail_issue_38701 = xfail_test(reason="RuntimeError: unsupported element type: STRING")
xfail_issue_38706 = xfail_test(reason="RuntimeError: output_3.0 has zero dimension which is not allowed")
xfail_issue_38708 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Slice): y>': "
                                      "Axes input must be constant")
xfail_issue_38710 = xfail_test(reason="RuntimeError: roi has zero dimension which is not allowed")
xfail_issue_38713 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ai.onnx.preview.training.Momentum")
xfail_issue_43742 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "If")
xfail_issue_45457 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v5::Loop"
                                      "Not constant termination condition body output is not supported")
xfail_issue_38722 = xfail_test(reason="RuntimeError: While validating ONNX nodes MatMulInteger"
                                      "and QLinearMatMul"
                                      "Input0 scale and input0 zero point shape must be same and 1")
xfail_issue_38723 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "QLinearConv")
xfail_issue_38724 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Resize): Y>':"
                                      "tf_crop_and_resize - this type of coordinate transformation mode"
                                      "is not supported. Choose one of the following modes:"
                                      "tf_half_pixel_for_nn, asymmetric, align_corners, pytorch_half_pixel,"
                                      "half_pixel")
xfail_issue_38725 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Loop):"
                                      "value info has no element type specified")
xfail_issue_38726 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "LessOrEqual")
xfail_issue_38732 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ConvInteger")
xfail_issue_38734 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ai.onnx.preview.training.Adam")
xfail_issue_38735 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ai.onnx.preview.training.Adagrad")
xfail_issue_48052 = xfail_test(reason="Dropout op is not supported in traning mode")
xfail_issue_45180 = xfail_test(reason="RuntimeError: Unsupported dynamic op: ReduceSum")
xfail_issue_44848 = xfail_test(reason="E   Unsupported dynamic op: Range")
xfail_issue_44851 = xfail_test(reason="E   Unsupported dynamic op: Broadcast")
xfail_issue_44854 = xfail_test(reason="E   Unsupported dynamic op: VariadicSplit")
xfail_issue_44858 = xfail_test(reason="E   Unsupported dynamic op: Unsqueeze")
xfail_issue_44956 = xfail_test(reason="E   Unsupported dynamic op: Loop")
xfail_issue_44957 = xfail_test(reason="E   Unsupported dynamic op: NonZero")
xfail_issue_44958 = xfail_test(reason="E   Unsupported dynamic op: Interpolate")
xfail_issue_44965 = xfail_test(reason="E   RuntimeError: value info has no element")
xfail_issue_44968 = xfail_test(reason="E   Unsupported dynamic op: Squeeze")
xfail_issue_44976 = xfail_test(reason="E   RuntimeError: Quantize layer with name:"
                                      "FakeQuantize_xxx has non const input on 1 port")
xfail_issue_46762 = xfail_test(reason="Incorrect result of Minimum op if uint data type is used")
xfail_issue_47323 = xfail_test(reason="RuntimeError: The plugin does not support FP64")
xfail_issue_47337 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::OneHot")
xfail_issue_33593 = xfail_test(reason="Current implementation of MaxPool doesn't support indices output")
xfail_issue_55760 = xfail_test(reason="RuntimeError: Reversed axis have axes above the source space shape")

# Model MSFT issues:
xfail_issue_37957 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "com.microsoft.CropAndResize, com.microsoft.GatherND,"
                                      "com.microsoft.Pad, com.microsoft.Range")
xfail_issue_39669 = xfail_test(reason="AssertionError: This model has no test data")
xfail_issue_40686 = xfail_test(reason="NANs as results")
xfail_issue_36534 = xfail_test(reason="RuntimeError: node input index is out of range")
xfail_issue_36536 = xfail_test(reason="RuntimeError: can't protect")
xfail_issue_36538 = xfail_test(reason="RuntimeError: Check 'PartialShape::broadcast_merge_into( pshape, "
                                      "node->get_input_partial_shape(i), autob)' failed at "
                                      "/openvino/ngraph/src/ngraph/op/util/elementwise_args.cpp:48:")
xfail_issue_39656 = xfail_test(reason="RuntimeError: Reshape reshaped has dynamic second input!")
xfail_issue_39658 = xfail_test(reason="RuntimeError: Tile operation has a form that is not supported."
                                      " z should be converted to TileIE operation.")
xfail_issue_39659 = xfail_test(reason="RuntimeError: Broadcast operation has a form that is not supported."
                                      " y should be converted to Tile operation.")
xfail_issue_45344 = xfail_test(reason="Unsupported dynamic ops: v3::NonMaxSuppressionIE3")
xfail_issue_39662 = xfail_test(reason="RuntimeError: 'ScatterElementsUpdate' layer with name 'y' have "
                                      "indices value that points to non-existing output tensor element")


xfail_issue_37973 = xfail_test(reason="TF Inception V2 - AssertionError: zoo models results mismatch")
xfail_issue_47430 = xfail_test(reason="FCN ResNet models - AssertionError: zoo models results mismatch")
xfail_issue_47495 = xfail_test(reason="BertSquad-10 from MSFT - AssertionError: zoo models results mismatch")
xfail_issue_49207 = xfail_test(reason="Function references undeclared parameters")
xfail_issue_48145 = xfail_test(reason="BertSquad-8 - AssertionError: Items are not equal: ACTUAL: 4 "
                                      "DESIRED: 3")
xfail_issue_48190 = xfail_test(reason="RobertaBase-11 - AssertionError: Items are not equal: "
                                      "ACTUAL: dtype('float64') DESIRED: dtype('float32')")
xfail_issue_49750 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v4::Interpolate")
xfail_issue_49752 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::Pad")
xfail_issue_49753 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::StridedSlice")
xfail_issue_49754 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::TopKIE")
xfail_issue_52463 = xfail_test(reason="test_operator_add_size1_singleton_broadcast_cpu - "
                                      "Not equal to tolerance")
xfail_issue_45432 = xfail_test(reason="Einsum is not implemented in CPU plugin.")
xfail_issue_onnx_models_140 = xfail_test(reason="https://github.com/onnx/models/issues/140")
