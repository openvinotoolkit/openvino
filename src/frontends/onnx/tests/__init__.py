# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

# test.BACKEND_NAME is a configuration variable determining which
# OV backend tests will use. It's set during pytest configuration time.
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
xfail_dynamic_rank = xfail_test(reason="Dynamic rank")
xfail_accuracy = xfail_test(reason="Accuracy")
xfail_issue_69444 = xfail_test(reason="ONNX Resize - AssertionError: Mismatched elements.")
skip_issue_67415 = pytest.mark.skip(reason="RuntimeError: Unsupported data type for when filling blob!")
xfail_issue_67415 = xfail_test(reason="RuntimeError: Unsupported data type for when filling blob!")
xfail_issue_33488 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations: "
                                      "MaxUnpool")
skip_issue_38084 = pytest.mark.skip(reason="Aborted (core dumped) Assertion "
                                           "`(layer->get_output_partial_shape(i).is_static())' failed.")
xfail_issue_33596 = xfail_test(reason="RuntimeError: OV does not support different sequence operations: "
                                      "ConcatFromSequence, SequenceConstruct, SequenceAt, SplitToSequence, "
                                      "SequenceEmpty, SequenceInsert, SequenceErase, SequenceLength ")
xfail_issue_33606 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations: "
                                      "Det")
xfail_issue_33651 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations: "
                                      "TfIdfVectorizer")
xfail_issue_33581 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations: "
                                      "GatherElements")
xfail_issue_90649 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations:"
                                      "DFT, LayerNormalization, "
                                      "MelWeightMatrix, SequenceMap, STFT")
xfail_issue_35923 = xfail_test(reason="RuntimeError: PReLU without weights is not supported")
xfail_issue_38091 = xfail_test(reason="AssertionError: Mismatched elements")
xfail_issue_38699 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations: "
                                      "ai.onnx.preview.training.Gradient")
xfail_issue_38701 = xfail_test(reason="RuntimeError: unsupported element type: STRING")
xfail_issue_38706 = xfail_test(reason="RuntimeError: output_3.0 has zero dimension which is not allowed")
xfail_issue_38708 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Slice): y>': "
                                      "Axes input must be constant")
skip_bitwise_ui64 = pytest.mark.skip(reason="AssertionError: Not equal to tolerance rtol=0.001, atol=1e-07")
xfail_issue_99949 = xfail_test(reason="Bitwise operators are not supported")
xfail_issue_99950 = xfail_test(reason="CenterCropPad func is not supported")
xfail_issue_99952 = xfail_test(reason="Col2Im operator is not supported")
xfail_issue_99954 = xfail_test(reason="Constant Pad - RuntimeError: Shape inference of Reference node with name y failed")
xfail_issue_99955 = xfail_test(reason="GroupNorm is not supported")
xfail_issue_99957 = xfail_test(reason="LayerNorm - RuntimeError: While validating node '<Node(Reshape): Mean>'")
xfail_issue_99960 = xfail_test(reason="MVN - Results mismatch")
xfail_issue_99961 = xfail_test(reason="Optional has/get element operators are not supported)'")
xfail_issue_99962 = pytest.mark.skip(reason="ReduceL1 - Unrecognized attribute: axes for operator ReduceL1")
xfail_issue_99968 = xfail_test(reason="ReduceL1 - Results mismatch or unsupported ReduceSum with "
                                      "dynamic rank by CPU plugin")
xfail_issue_99969 = xfail_test(reason="Resize - Results mismatch / "
                                      "RuntimeError: While validating ONNX node '<Node(Resize): Y>' / "
                                      "RuntimeError: Check '(false)' failed at onnx/frontend/src/op/resize.cpp")
xfail_issue_99970 = xfail_test(reason="Scatter and ScatterND - RuntimeError: Check '(reduction == none)' failed at "
                                      "src/frontends/onnx/frontend/src/op/scatter_elements.cpp OR at "
                                      "src/frontends/onnx/frontend/src/op/scatter_nd")
xfail_issue_99973 = xfail_test(reason="Split -  RuntimeError: While validating ONNX node "
                                      "'<Node(Split): output_1, output_2, output_3, output_4>'")
xfail_issue_38710 = xfail_test(reason="RuntimeError: data has zero dimension which is not allowed")
xfail_issue_38713 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations: "
                                      "ai.onnx.preview.training.Momentum")
xfail_issue_38724 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Resize): Y>': "
                                      "tf_crop_and_resize - this type of coordinate transformation mode "
                                      "is not supported. Choose one of the following modes: "
                                      "tf_half_pixel_for_nn, asymmetric, align_corners, pytorch_half_pixel, "
                                      "half_pixel")
xfail_issue_38725 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Loop): "
                                      "value info has no element type specified")
xfail_issue_38734 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations: "
                                      "ai.onnx.preview.training.Adam")
xfail_issue_38735 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations: "
                                      "ai.onnx.preview.training.Adagrad")
xfail_issue_48052 = xfail_test(reason="Dropout op is not supported in traning mode")
xfail_issue_44851 = xfail_test(reason="Expected: Unsupported dynamic op: Broadcast")
xfail_issue_44858 = xfail_test(reason="Expected: Unsupported dynamic op: Unsqueeze")
xfail_issue_44957 = xfail_test(reason="Expected: Unsupported dynamic op: NonZero")
xfail_issue_44958 = xfail_test(reason="Expected: Unsupported dynamic op: Interpolate")
xfail_issue_44965 = xfail_test(reason="Expected: RuntimeError: value info has no element")
xfail_issue_47323 = xfail_test(reason="RuntimeError: The plugin does not support FP64")
xfail_issue_73538 = xfail_test(reason="OneHot: Unsupported negative indices, "
                                      "AssertionError: Mismatched elements.")
# Model MSFT issues:
xfail_issue_37957 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations: "
                                      "com.microsoft.CropAndResize, com.microsoft.GatherND, "
                                      "com.microsoft.Pad, com.microsoft.Range")
xfail_issue_39669 = xfail_test(reason="AssertionError: This model has no test data")
xfail_issue_36534 = xfail_test(reason="RuntimeError: node input index is out of range")
xfail_issue_36536 = xfail_test(reason="RuntimeError: can't protect")
xfail_issue_36538 = xfail_test(reason="RuntimeError: Check 'PartialShape::broadcast_merge_into( pshape, "
                                      "node->get_input_partial_shape(i), autob)' failed at ")
skip_issue_39658 = pytest.mark.skip(reason="RuntimeError: Tile operation has a form that is not supported."
                                           " z should be converted to TileIE operation.")


xfail_issue_37973 = xfail_test(reason="TF Inception V2 - AssertionError: zoo models results mismatch")
xfail_issue_47495 = xfail_test(reason="BertSquad-10 from MSFT - AssertionError: zoo models results mismatch")
xfail_issue_49207 = xfail_test(reason="Model references undeclared parameters")
xfail_issue_48145 = xfail_test(reason="BertSquad-8 - AssertionError: Items are not equal: ACTUAL: 4 "
                                      "DESIRED: 3")
xfail_issue_48190 = xfail_test(reason="RobertaBase-11 - AssertionError: Items are not equal: "
                                      "ACTUAL: dtype('float64') DESIRED: dtype('float32')")
xfail_issue_49752 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::Pad")
xfail_issue_49753 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::StridedSlice")
xfail_issue_49754 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::TopKIE")
xfail_issue_52463 = xfail_test(reason="test_operator_add_size1_singleton_broadcast_cpu - "
                                      "Not equal to tolerance")
xfail_issue_58033 = xfail_test(reason="Einsum operation misses support for complex ellipsis equations")
xfail_issue_58676 = xfail_test(reason="AssertionError: Not equal to tolerance rtol=0.001, atol=1e-07")
skip_issue_58676 = pytest.mark.skip(reason="AssertionError: Not equal to tolerance rtol=0.001, atol=1e-07")
xfail_issue_onnx_models_140 = xfail_test(reason="https://github.com/onnx/models/issues/140")

xfail_issue_63033 = xfail_test(reason="BatchNormalization: Training mode is not supported")
xfail_issue_63036 = xfail_test(reason="Changes in ConvTranspose padding")
xfail_issue_63043 = xfail_test(reason="Recurrent node expects constants as W, R, B inputs.")

skip_rng_tests = pytest.mark.skip(reason="Tests use random number generator with no seed.")
xfail_issue_63137 = xfail_test(reason="Unsupported operations: OptionalHasElement, OptionalGetElement")
xfail_issue_68212 = xfail_test(reason="Unsupported reading model with bytes streams")

xfail_issue_78843 = xfail_test(reason="Missing reference output files for ssd mobilenet models")

xfail_issue_81976 = xfail_test(reason="RuntimeError: z node not found in OV cache")
xfail_issue_82038 = xfail_test(reason="ScatterElements, ScatterND, AssertionError: Result mismatch")
xfail_issue_82039 = xfail_test(reason="Unsupported data type Optional, RuntimeError: [ NOT_IMPLEMENTED ] "
                                      "CPU plugin: Input image format UNSPECIFIED is not supported yet...")

xfail_issue_86911 = xfail_test(reason="LSTM_Seq_len_unpacked - AssertionError: zoo models results mismatch")
xfail_issue_101965 = xfail_test(reason="Mismatch with numpy-based expected results.")
xfail_issue_113506 = xfail_test(reason="Unsupported operation of type: LSTMSequence Node expects 7 inputs. Actual: 8")

skip_dynamic_model = pytest.mark.skip(reason="CPU plug-in can't load a model with dynamic output shapes via legacy API")

# ONNX 1.14
xfail_issue_119896 = xfail_test(reason="Unsupported element type: FLOAT8")
xfail_issue_119900 = xfail_test(reason="While validating ONNX node '<Node(Resize): Y>': "
                                       "half_pixel_symmetric - this type of coordinate transformation mode "
                                       "is not supported. Choose one of the following modes: "
                                       "tf_half_pixel_for_nn, asymmetric, align_corners, pytorch_half_pixel, "
                                       "half_pixel")
xfail_issue_119903 = xfail_test(reason="DeformConv operation is not supported")
xfail_issue_119906 = xfail_test(reason="LpPool operation is not supported")
xfail_issue_119919 = xfail_test(reason="While validating ONNX node '<Node(Pad): y>': Unsupported padding mode: [wrap]")
xfail_issue_119922 = xfail_test(reason="ai.onnx.ml operators domain isn't supported")
xfail_issue_119925 = xfail_test(reason="AveragePool AssertionError: Not equal to tolerance rtol=0.001, atol=1e-07")
xfail_issue_119926 = xfail_test(reason="ROIAlign AssertionError: Not equal to tolerance rtol=0.001, atol=1e-07")

# ONNX 1.15
xfail_issue_125485 = xfail_test(reason="AffineGrid operation is not supported")
xfail_issue_125488 = xfail_test(reason="ImageDecoder operation is not supported")
skip_issue_125487 = pytest.mark.skip(reason="GridSample doesn't support cubic and linear modes, and 4D tensor") # Need to enable after bumping to 1.15
skip_issue_125489 = pytest.mark.skip(reason="IsInf changed behavior since opset-20") # Need to enable after opset-20 will be released
skip_issue_124587 = pytest.mark.skip(reason="Fail on new macos machines")
xfail_issue_125491 = xfail_test(reason="AveragePool mismatch with differences in shapes")
xfail_issue_125492 = xfail_test(reason="DFT mismatch")
xfail_issue_125493 = xfail_test(reason="Reduce* mismatch")
xfail_issue_122776 = xfail_test(reason="test_mish_expanded_cpu - "
                                       "Not equal to tolerance")
xfail_issue_122775 = xfail_test(reason="test_resize_downsample_scales_linear_cpu - "
                                       "Not equal to tolerance")
skip_issue_127649 = pytest.mark.skip(reason="Not equal to tolerance rtol=0.001, atol=1e-07 - "
                                             "Mismatched elements: 1 / 1000 (0.1%)")

# ONNX 1.16
skip_misalignment = pytest.mark.skip(reason="Misalignment between onnx versions") # Need to enable after bumping to 1.16
xfail_issue_139934 = xfail_test(reason = "Int4 isn't supported")
xfail_issue_139936 = xfail_test(reason = "MaxPool accuracy fails")
xfail_issue_139937 = xfail_test(reason = "GroupNorm, QLinearMatMul, DequantizeLinear translation failed")
xfail_issue_139938 = xfail_test(reason = "QLinearMatMul accuracy fails")

# ONNX 1.17
skip_issue_119896 = pytest.mark.skip(reason="Unsupported element type: FLOAT8")
