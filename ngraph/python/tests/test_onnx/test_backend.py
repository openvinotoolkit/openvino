# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import onnx.backend.test

from tests.test_onnx.utils import OpenVinoOnnxBackend
from tests import (BACKEND_NAME,
                   xfail_issue_33540,
                   xfail_issue_34314,
                   xfail_issue_35926,
                   xfail_issue_1,
                   xfail_issue_33616,
                   xfail_issue_2,
                   xfail_issue_3,
                   xfail_issue_4,
                   xfail_issue_35893,
                   xfail_issue_35923,
                   xfail_issue_35914,
                   xfail_issue_36483,
                   xfail_issue_34323,
                   xfail_issue_35915,
                   xfail_issue_5,
                   xfail_issue_36476,
                   xfail_issue_36478,
                   xfail_issue_36437,
                   xfail_issue_6,
                   xfail_issue_7,
                   xfail_issue_8,
                   xfail_issue_9)


def expect_fail(test_case_path, xfail):  # type: (str) -> None
    """Mark the test as expected to fail."""
    module_name, test_name = test_case_path.split(".")
    module = globals().get(module_name)
    if hasattr(module, test_name):
        xfail(getattr(module, test_name))
    else:
        logging.getLogger().warning("Could not mark test as XFAIL, not found: %s", test_case_path)


OpenVinoOnnxBackend.backend_name = BACKEND_NAME

# This is a pytest magic variable to load extra plugins
# Uncomment the line below to enable the ONNX compatibility report
# pytest_plugins = "onnx.backend.test.report",

# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(OpenVinoOnnxBackend, __name__)

skip_tests_general = [
    # Big model tests (see test_zoo_models.py):
    "test_bvlc_alexnet",
    "test_densenet121",
    "test_inception_v1",
    "test_inception_v2",
    "test_resnet50",
    "test_shufflenet",
    "test_squeezenet",
    "test_vgg19",
    "test_zfnet512",
]

for test in skip_tests_general:
    backend_test.exclude(test)

# NOTE: ALL backend_test.exclude CALLS MUST BE PERFORMED BEFORE THE CALL TO globals().update

OnnxBackendNodeModelTest = None
OnnxBackendSimpleModelTest = None
OnnxBackendPyTorchOperatorModelTest = None
OnnxBackendPyTorchConvertedModelTest = None
globals().update(backend_test.enable_report().test_cases)

tests_xfail_new = [
    (xfail_issue_34314,
        "OnnxBackendNodeModelTest.test_rnn_seq_length_cpu",
        "OnnxBackendNodeModelTest.test_simple_rnn_defaults_cpu",
        "OnnxBackendNodeModelTest.test_simple_rnn_with_initial_bias_cpu"),
    (xfail_issue_33540,
        "OnnxBackendNodeModelTest.test_gru_defaults_cpu",
        "OnnxBackendNodeModelTest.test_gru_seq_length_cpu",
        "OnnxBackendNodeModelTest.test_gru_with_initial_bias_cpu"),
    (xfail_issue_35926,
        "OnnxBackendNodeModelTest.test_expand_dim_changed_cpu",
        "OnnxBackendNodeModelTest.test_expand_dim_unchanged_cpu",
        "OnnxBackendSimpleModelTest.test_expand_shape_model1_cpu",
        "OnnxBackendSimpleModelTest.test_expand_shape_model2_cpu",
        "OnnxBackendSimpleModelTest.test_expand_shape_model3_cpu",
        "OnnxBackendSimpleModelTest.test_expand_shape_model4_cpu",
        "OnnxBackendNodeModelTest.test_reshape_extended_dims_cpu",
        "OnnxBackendNodeModelTest.test_reshape_negative_dim_cpu",
        "OnnxBackendNodeModelTest.test_reshape_one_dim_cpu",
        "OnnxBackendNodeModelTest.test_reshape_reduced_dims_cpu",
        "OnnxBackendNodeModelTest.test_reshape_negative_extended_dims_cpu",
        "OnnxBackendNodeModelTest.test_reshape_reordered_all_dims_cpu",
        "OnnxBackendNodeModelTest.test_reshape_reordered_last_dims_cpu",
        "OnnxBackendNodeModelTest.test_reshape_zero_and_negative_dim_cpu",
        "OnnxBackendNodeModelTest.test_reshape_zero_dim_cpu",
        "OnnxBackendNodeModelTest.test_tile_cpu",
        "OnnxBackendNodeModelTest.test_tile_precomputed_cpu",
        "OnnxBackendNodeModelTest.test_constantofshape_float_ones_cpu",
        "OnnxBackendNodeModelTest.test_constantofshape_int_zeros_cpu",
        "OnnxBackendNodeModelTest.test_scatter_with_axis_cpu",
        "OnnxBackendNodeModelTest.test_scatter_without_axis_cpu",
        "OnnxBackendNodeModelTest.test_nonmaxsuppression_center_point_box_format_cpu",
        "OnnxBackendNodeModelTest.test_nonmaxsuppression_flipped_coordinates_cpu",
        "OnnxBackendNodeModelTest.test_nonmaxsuppression_identical_boxes_cpu",
        "OnnxBackendNodeModelTest.test_nonmaxsuppression_limit_output_size_cpu",
        "OnnxBackendNodeModelTest.test_nonmaxsuppression_single_box_cpu",
        "OnnxBackendNodeModelTest.test_nonmaxsuppression_suppress_by_IOU_and_scores_cpu",
        "OnnxBackendNodeModelTest.test_nonmaxsuppression_suppress_by_IOU_cpu",
        "OnnxBackendNodeModelTest.test_nonmaxsuppression_two_batches_cpu",
        "OnnxBackendNodeModelTest.test_nonmaxsuppression_two_classes_cpu",
        "OnnxBackendNodeModelTest.test_slice_default_axes_cpu",
        "OnnxBackendNodeModelTest.test_roialign_cpu",
        "OnnxBackendNodeModelTest.test_scatter_elements_with_axis_cpu",
        "OnnxBackendNodeModelTest.test_scatter_elements_with_negative_indices_cpu",
        "OnnxBackendNodeModelTest.test_scatter_elements_without_axis_cpu",
        "OnnxBackendNodeModelTest.test_constant_pad_cpu",
        "OnnxBackendNodeModelTest.test_edge_pad_cpu",
        "OnnxBackendNodeModelTest.test_reflect_pad_cpu",
        "OnnxBackendNodeModelTest.test_top_k_cpu",
        "OnnxBackendNodeModelTest.test_top_k_negative_axis_cpu",
        "OnnxBackendNodeModelTest.test_top_k_smallest_cpu",
        "OnnxBackendNodeModelTest.test_where_long_example_cpu",
        "OnnxBackendNodeModelTest.test_gather_0_cpu",
        "OnnxBackendNodeModelTest.test_gather_1_cpu",
        "OnnxBackendNodeModelTest.test_mod_int64_fmod_cpu",
        "OnnxBackendNodeModelTest.test_reversesequence_batch_cpu",
        "OnnxBackendNodeModelTest.test_reversesequence_time_cpu",
        "OnnxBackendPyTorchOperatorModelTest.test_operator_non_float_params_cpu",
        "OnnxBackendPyTorchConvertedModelTest.test_Embedding_cpu",
        "OnnxBackendPyTorchConvertedModelTest.test_Embedding_sparse_cpu",
        "OnnxBackendNodeModelTest.test_constantofshape_int_shape_zero_cpu",
        "OnnxBackendNodeModelTest.test_max_int64_cpu",
        "OnnxBackendNodeModelTest.test_pow_types_int64_int64_cpu",
        "OnnxBackendNodeModelTest.test_min_int64_cpu",
        "OnnxBackendNodeModelTest.test_gather_negative_indices_cpu"),
    (xfail_issue_1,
        "OnnxBackendNodeModelTest.test_nonzero_example_cpu",
        "OnnxBackendNodeModelTest.test_range_int32_type_negative_delta_cpu",
        "OnnxBackendNodeModelTest.test_range_float_type_positive_delta_cpu"),
    (xfail_issue_33616,
        "OnnxBackendNodeModelTest.test_maxpool_2d_ceil_cpu",
        "OnnxBackendNodeModelTest.test_maxpool_2d_dilations_cpu",
        "OnnxBackendNodeModelTest.test_averagepool_2d_ceil_cpu"),
    (xfail_issue_2,
        "OnnxBackendNodeModelTest.test_upsample_nearest_cpu"),
    (xfail_issue_3,
        "OnnxBackendNodeModelTest.test_dynamicquantizelinear_min_adjusted_expanded_cpu",
        "OnnxBackendNodeModelTest.test_dynamicquantizelinear_expanded_cpu",
        "OnnxBackendNodeModelTest.test_dynamicquantizelinear_max_adjusted_expanded_cpu",
        "OnnxBackendNodeModelTest.test_quantizelinear_cpu",
     ),
    (xfail_issue_4,
        "OnnxBackendNodeModelTest.test_convtranspose_1d_cpu"),
    (xfail_issue_35893,
        "OnnxBackendNodeModelTest.test_convtranspose_3d_cpu",
        "OnnxBackendNodeModelTest.test_convtranspose_cpu",
        "OnnxBackendNodeModelTest.test_convtranspose_dilations_cpu",
        "OnnxBackendNodeModelTest.test_convtranspose_kernel_shape_cpu",
        "OnnxBackendNodeModelTest.test_convtranspose_output_shape_cpu",
        "OnnxBackendNodeModelTest.test_convtranspose_pad_cpu",
        "OnnxBackendNodeModelTest.test_convtranspose_pads_cpu",
        "OnnxBackendNodeModelTest.test_convtranspose_with_kernel_cpu",
        "OnnxBackendNodeModelTest.test_instancenorm_example_cpu",
        "OnnxBackendNodeModelTest.test_basic_conv_without_padding_cpu",
        "OnnxBackendNodeModelTest.test_batchnorm_epsilon_cpu",
        "OnnxBackendNodeModelTest.test_batchnorm_example_cpu",
        "OnnxBackendNodeModelTest.test_conv_with_strides_and_asymmetric_padding_cpu",
        "OnnxBackendNodeModelTest.test_conv_with_strides_no_padding_cpu",
        "OnnxBackendNodeModelTest.test_conv_with_strides_padding_cpu",
        "OnnxBackendNodeModelTest.test_instancenorm_epsilon_cpu",
        "OnnxBackendNodeModelTest.test_basic_conv_with_padding_cpu"),
    (xfail_issue_35923,
        "OnnxBackendNodeModelTest.test_prelu_broadcast_cpu",
        "OnnxBackendNodeModelTest.test_prelu_example_cpu"),
    (xfail_issue_35914,
        "OnnxBackendNodeModelTest.test_dequantizelinear_cpu",
        "OnnxBackendNodeModelTest.test_pow_bcast_scalar_cpu",
        "OnnxBackendNodeModelTest.test_clip_example_cpu",
        "OnnxBackendNodeModelTest.test_clip_inbounds_cpu",
        "OnnxBackendNodeModelTest.test_clip_outbounds_cpu",
        "OnnxBackendNodeModelTest.test_clip_default_int8_min_cpu",
        "OnnxBackendNodeModelTest.test_clip_default_int8_max_cpu",
        "OnnxBackendNodeModelTest.test_gemm_default_scalar_bias_cpu",
        "OnnxBackendNodeModelTest.test_clip_cpu",
        "OnnxBackendNodeModelTest.test_clip_default_max_cpu",
        "OnnxBackendNodeModelTest.test_clip_default_min_cpu",
        "OnnxBackendNodeModelTest.test_clip_splitbounds_cpu"),
    (xfail_issue_36483,
        "OnnxBackendNodeModelTest.test_ceil_cpu",
        "OnnxBackendNodeModelTest.test_ceil_example_cpu"),
    (xfail_issue_34323,
        "OnnxBackendNodeModelTest.test_constant_cpu",
        "OnnxBackendNodeModelTest.test_dropout_default_cpu",
        "OnnxBackendNodeModelTest.test_eyelike_populate_off_main_diagonal_cpu",
        "OnnxBackendNodeModelTest.test_eyelike_without_dtype_cpu",
        "OnnxBackendNodeModelTest.test_max_one_input_cpu",
        "OnnxBackendNodeModelTest.test_min_one_input_cpu",
        "OnnxBackendNodeModelTest.test_shape_cpu",
        "OnnxBackendNodeModelTest.test_shape_example_cpu",
        "OnnxBackendNodeModelTest.test_size_cpu",
        "OnnxBackendNodeModelTest.test_size_example_cpu",
        "OnnxBackendNodeModelTest.test_sum_one_input_cpu",
        "OnnxBackendNodeModelTest.test_dropout_default_ratio_cpu",
        "OnnxBackendNodeModelTest.test_dropout_default_old_cpu",
        "OnnxBackendNodeModelTest.test_dropout_random_old_cpu",
        "OnnxBackendNodeModelTest.test_training_dropout_default_cpu",
        "OnnxBackendNodeModelTest.test_training_dropout_zero_ratio_cpu",
        "OnnxBackendNodeModelTest.test_training_dropout_cpu"),
    (xfail_issue_35915,
        "OnnxBackendNodeModelTest.test_equal_bcast_cpu",
        "OnnxBackendNodeModelTest.test_equal_cpu",
        "OnnxBackendNodeModelTest.test_min_int16_cpu",
        "OnnxBackendNodeModelTest.test_min_uint8_cpu"),
    (xfail_issue_5,
        "OnnxBackendNodeModelTest.test_lstm_defaults_cpu",
        "OnnxBackendNodeModelTest.test_lstm_with_initial_bias_cpu",
        "OnnxBackendNodeModelTest.test_lstm_with_peepholes_cpu"),
    (xfail_issue_36476,
        "OnnxBackendNodeModelTest.test_max_uint32_cpu",
        "OnnxBackendNodeModelTest.test_min_uint32_cpu"),
    (xfail_issue_36478,
        "OnnxBackendNodeModelTest.test_max_uint64_cpu",
        "OnnxBackendNodeModelTest.test_min_uint64_cpu"),
    (xfail_issue_36437,
        "OnnxBackendNodeModelTest.test_argmax_default_axis_example_cpu",
        "OnnxBackendNodeModelTest.test_argmax_default_axis_random_cpu",
        "OnnxBackendNodeModelTest.test_argmax_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_argmax_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_argmax_negative_axis_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_argmax_negative_axis_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_argmin_default_axis_example_cpu",
        "OnnxBackendNodeModelTest.test_argmin_default_axis_random_cpu",
        "OnnxBackendNodeModelTest.test_argmin_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_argmin_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_argmin_negative_axis_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_argmin_negative_axis_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_argmax_keepdims_random_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmax_negative_axis_keepdims_example_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmax_keepdims_example_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmin_default_axis_random_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmin_default_axis_example_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmax_default_axis_random_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmax_default_axis_example_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmax_negative_axis_keepdims_random_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmin_negative_axis_keepdims_random_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmin_negative_axis_keepdims_example_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmin_keepdims_example_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmin_keepdims_random_select_last_index_cpu"),
    (xfail_issue_6,
        "OnnxBackendPyTorchConvertedModelTest.test_GLU_cpu"),
    (xfail_issue_7,
        "OnnxBackendPyTorchConvertedModelTest.test_GLU_dim_cpu"),
    (xfail_issue_8,
        "OnnxBackendNodeModelTest.test_not_2d_cpu",
        "OnnxBackendNodeModelTest.test_not_3d_cpu",
        "OnnxBackendNodeModelTest.test_not_4d_cpu",
        "OnnxBackendNodeModelTest.test_or2d_cpu",
        "OnnxBackendNodeModelTest.test_or3d_cpu",
        "OnnxBackendNodeModelTest.test_or4d_cpu",
        "OnnxBackendNodeModelTest.test_or_bcast3v1d_cpu",
        "OnnxBackendNodeModelTest.test_or_bcast3v2d_cpu",
        "OnnxBackendNodeModelTest.test_or_bcast4v2d_cpu",
        "OnnxBackendNodeModelTest.test_or_bcast4v3d_cpu",
        "OnnxBackendNodeModelTest.test_or_bcast4v4d_cpu",
        "OnnxBackendNodeModelTest.test_xor2d_cpu",
        "OnnxBackendNodeModelTest.test_xor3d_cpu",
        "OnnxBackendNodeModelTest.test_xor4d_cpu",
        "OnnxBackendNodeModelTest.test_xor_bcast3v1d_cpu",
        "OnnxBackendNodeModelTest.test_xor_bcast3v2d_cpu",
        "OnnxBackendNodeModelTest.test_xor_bcast4v2d_cpu",
        "OnnxBackendNodeModelTest.test_xor_bcast4v3d_cpu",
        "OnnxBackendNodeModelTest.test_xor_bcast4v4d_cpu",
        "OnnxBackendNodeModelTest.test_greater_equal_expanded_cpu",
        "OnnxBackendNodeModelTest.test_less_equal_bcast_expanded_cpu",
        "OnnxBackendNodeModelTest.test_max_int16_cpu",
        "OnnxBackendNodeModelTest.test_max_uint16_cpu",
        "OnnxBackendNodeModelTest.test_less_equal_expanded_cpu",
        "OnnxBackendNodeModelTest.test_max_int8_cpu",
        "OnnxBackendNodeModelTest.test_max_uint8_cpu",
        "OnnxBackendNodeModelTest.test_maxpool_2d_uint8_cpu",
        "OnnxBackendNodeModelTest.test_min_float16_cpu",
        "OnnxBackendNodeModelTest.test_greater_equal_bcast_expanded_cpu",
        "OnnxBackendNodeModelTest.test_min_uint16_cpu",
        "OnnxBackendNodeModelTest.test_max_float16_cpu",
        "OnnxBackendNodeModelTest.test_min_int8_cpu",
        "OnnxBackendNodeModelTest.test_greater_bcast_cpu",
        "OnnxBackendNodeModelTest.test_greater_cpu",
        "OnnxBackendNodeModelTest.test_less_bcast_cpu",
        "OnnxBackendNodeModelTest.test_less_cpu",
        "OnnxBackendNodeModelTest.test_mod_mixed_sign_float16_cpu",
        "OnnxBackendNodeModelTest.test_argmax_no_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_argmax_no_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_argmin_no_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_argmax_no_keepdims_random_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmax_no_keepdims_example_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmin_no_keepdims_example_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_argmin_no_keepdims_random_select_last_index_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT_to_FLOAT16_cpu",
        "OnnxBackendNodeModelTest.test_and3d_cpu",
        "OnnxBackendNodeModelTest.test_and4d_cpu",
        "OnnxBackendNodeModelTest.test_and_bcast3v1d_cpu",
        "OnnxBackendNodeModelTest.test_and_bcast3v2d_cpu",
        "OnnxBackendNodeModelTest.test_and_bcast4v2d_cpu",
        "OnnxBackendNodeModelTest.test_and_bcast4v3d_cpu",
        "OnnxBackendNodeModelTest.test_and_bcast4v4d_cpu",
        "OnnxBackendNodeModelTest.test_argmin_no_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_clip_default_int8_inbounds_cpu",
        "OnnxBackendNodeModelTest.test_and2d_cpu"),
    (xfail_issue_9,
        "OnnxBackendNodeModelTest.test_round_cpu",
        "OnnxBackendNodeModelTest.test_mvn_cpu",
        "OnnxBackendNodeModelTest.test_elu_example_cpu",
        "OnnxBackendNodeModelTest.test_logsoftmax_axis_0_cpu",
        "OnnxBackendNodeModelTest.test_logsoftmax_axis_1_cpu",
        "OnnxBackendNodeModelTest.test_logsoftmax_default_axis_cpu")
]

for test_group in tests_xfail_new:
    for test_case in test_group[1:]:
        expect_fail("{}".format(test_case), test_group[0])
