# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# The following tests contain expanded subgraphs representing an ONNX operator.
ONNX_FUNCTION_TESTS = {
    "celu_expanded": "Celu",
    "dynamicquantizelinear_expanded": "DynamicQuantizeLinear",
    "dynamicquantizelinear_max_adjusted_expanded": "DynamicQuantizeLinear",
    "dynamicquantizelinear_min_adjusted_expanded": "DynamicQuantizeLinear",
    "nllloss_NC_expanded": "NegativeLogLikelihoodLoss",
    "nllloss_NCd1_ii_expanded": "NegativeLogLikelihoodLoss",
    "nllloss_NCd1_mean_weight_negative_ii_expanded": "NegativeLogLikelihoodLoss",
    "nllloss_NCd1_weight_ii_expanded": "NegativeLogLikelihoodLoss",
    "nllloss_NCd1d2_expanded": "NegativeLogLikelihoodLoss",
    "nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded": "NegativeLogLikelihoodLoss",
    "nllloss_NCd1d2_with_weight_expanded": "NegativeLogLikelihoodLoss",
    "nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded": "NegativeLogLikelihoodLoss",
    "nllloss_NCd1d2d3_sum_weight_high_ii_expanded": "NegativeLogLikelihoodLoss",
    "nllloss_NCd1d2d3d4d5_none_no_weight_expanded": "NegativeLogLikelihoodLoss",
    "sce_NCd1_mean_weight_negative_ii_expanded": "SoftmaxCrossEntropyLoss",
    "sce_NCd1d2d3_none_no_weight_negative_ii_expanded": "SoftmaxCrossEntropyLoss",
    "sce_NCd1d2d3_sum_weight_high_ii_expanded": "SoftmaxCrossEntropyLoss",
    "sce_NCd1d2d3d4d5_mean_weight_expanded": "SoftmaxCrossEntropyLoss",
    "sce_NCd1d2d3d4d5_none_no_weight_expanded": "SoftmaxCrossEntropyLoss",
    "sce_mean_3d_expanded": "SoftmaxCrossEntropyLoss",
    "sce_mean_expanded": "SoftmaxCrossEntropyLoss",
    "sce_mean_no_weight_ii_3d_expanded": "SoftmaxCrossEntropyLoss",
    "sce_mean_no_weight_ii_4d_expanded": "SoftmaxCrossEntropyLoss",
    "sce_mean_no_weight_ii_expanded": "SoftmaxCrossEntropyLoss",
    "sce_mean_weight_expanded": "SoftmaxCrossEntropyLoss",
    "sce_mean_weight_ii_3d_expanded": "SoftmaxCrossEntropyLoss",
    "sce_mean_weight_ii_4d_expanded": "SoftmaxCrossEntropyLoss",
    "sce_mean_weight_ii_expanded": "SoftmaxCrossEntropyLoss",
    "sce_none_expanded": "SoftmaxCrossEntropyLoss",
    "sce_none_weights_expanded": "SoftmaxCrossEntropyLoss",
    "sce_sum_expanded": "SoftmaxCrossEntropyLoss",
    "softmax_axis_0_expanded": "Softmax",
    "softmax_axis_1_expanded": "Softmax",
    "softmax_axis_2_expanded": "Softmax",
    "softmax_default_axis_expanded": "Softmax",
    "softmax_example_expanded": "Softmax",
    "softmax_large_number_expanded": "Softmax",
    "softmax_negative_axis_expanded": "Softmax",
}

UNSUPPORTED_OPS = [
    "ConcatFromSequence",
    "MaxRoiPool",
    "Multinomial",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "SequenceAt",
    "SequenceErase",
    "SequenceLength",
]

# ops supported by OV but not tested by ONNX compliance tests
SUPPORTED_OPS = ["GlobalLpPool", "LpNormalization", "LpPool", "SpaceToDepth"]

# training related ops, should not be added to the report
DO_NOT_REPORT = ["Adagrad", "Adam", "Momentum"]


def find_tested_op(nodeid):
    """Returns a tested operator/function name for a given test."""
    for key in ONNX_FUNCTION_TESTS:
        if key in nodeid:
            return ONNX_FUNCTION_TESTS[key]

    return None
