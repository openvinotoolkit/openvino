# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform
import logging

import onnx.backend.test
from tests import (
    BACKEND_NAME,
    skip_rng_tests,
    xfail_issue_33488,
    xfail_issue_33581,
    xfail_issue_33596,
    xfail_issue_33606,
    xfail_issue_33651,
    xfail_issue_38091,
    xfail_issue_38699,
    xfail_issue_38701,
    xfail_issue_38706,
    xfail_issue_38710,
    xfail_issue_38713,
    xfail_issue_38724,
    xfail_issue_38734,
    xfail_issue_38735,
    skip_issue_39658,
    skip_issue_39658,
    skip_issue_58676,
    xfail_issue_44858,
    xfail_issue_44965,
    xfail_issue_45180,
    xfail_issue_47323,
    xfail_issue_73538,
    xfail_issue_48052,
    xfail_issue_52463,
    xfail_issue_58033,
    xfail_issue_63033,
    xfail_issue_63036,
    xfail_issue_63043,
    xfail_issue_63137,
    xfail_issue_63138,
    xfail_issue_69444,
    xfail_issue_81976,
    skip_segfault,
    xfail_issue_82038,
    xfail_issue_82039,
    xfail_issue_90649,
    xfail_issue_91151,
    xfail_issue_99949,
    xfail_issue_99950,
    xfail_issue_99952,
    xfail_issue_99954,
    xfail_issue_99955,
    xfail_issue_99957,
    xfail_issue_99958,
    xfail_issue_99959,
    xfail_issue_99960,
    xfail_issue_99961,
    xfail_issue_99968,
    xfail_issue_99969,
    xfail_issue_99970,
    xfail_issue_99972,
    xfail_issue_99973,
    xfail_issue_101965,
    xfail_issue_113506,
    skip_dynamic_model,
    xfail_issue_119896,
    xfail_issue_119900,
    xfail_issue_119903,
    xfail_issue_119906,
    xfail_issue_119919,
    xfail_issue_119922,
    xfail_issue_119925,
    xfail_issue_119926,
)
from tests.tests_python.utils.onnx_backend import OpenVinoTestBackend


def expect_fail(test_case_path, xfail):  # type: (str) -> None
    """Mark the test as expected to fail."""
    module_name, test_name = test_case_path.split(".")
    module = globals().get(module_name)
    if hasattr(module, test_name):
        xfail(getattr(module, test_name))
    else:
        logging.getLogger().warning(
            "Could not mark test as XFAIL, not found: %s", test_case_path,
        )


OpenVinoTestBackend.backend_name = BACKEND_NAME

"""This is a pytest magic variable to load extra plugins
Uncomment the line below to enable the ONNX compatibility report
pytest_plugins = "onnx.backend.test.report",
"""

# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(OpenVinoTestBackend, __name__)

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

tests_expected_to_fail = [
    (
        skip_issue_39658,
        "OnnxBackendNodeModelTest.test_tile_cpu",
    ),
    (
        xfail_issue_38091,
        "OnnxBackendNodeModelTest.test_dynamicquantizelinear_cpu",
        "OnnxBackendNodeModelTest.test_dynamicquantizelinear_expanded_cpu",
    ),
    (
        xfail_issue_52463,
        "OnnxBackendPyTorchOperatorModelTest.test_operator_add_size1_singleton_broadcast_cpu",
    ),
    (
        xfail_issue_47323,
        "OnnxBackendPyTorchOperatorModelTest.test_operator_add_broadcast_cpu",
        "OnnxBackendPyTorchOperatorModelTest.test_operator_addconstant_cpu",
        "OnnxBackendPyTorchOperatorModelTest.test_operator_add_size1_right_broadcast_cpu",
    ),
    (
        xfail_issue_38699,
        "OnnxBackendSimpleModelTest.test_gradient_of_add_and_mul_cpu",
        "OnnxBackendSimpleModelTest.test_gradient_of_add_cpu",
    ),
    (
        xfail_issue_33596,
        "OnnxBackendSimpleModelTest.test_sequence_model5_cpu",
        "OnnxBackendSimpleModelTest.test_sequence_model7_cpu",
        "OnnxBackendSimpleModelTest.test_sequence_model1_cpu",
        "OnnxBackendSimpleModelTest.test_sequence_model3_cpu",
        "OnnxBackendSimpleModelTest.test_sequence_model6_cpu",
        "OnnxBackendSimpleModelTest.test_sequence_model8_cpu",
        "OnnxBackendSimpleModelTest.test_sequence_model4_cpu",
        "OnnxBackendSimpleModelTest.test_sequence_model2_cpu",
        "OnnxBackendNodeModelTest.test_identity_sequence_cpu",
        "OnnxBackendNodeModelTest.test_if_seq_cpu",
        "OnnxBackendNodeModelTest.test_if_opt_cpu",  # Optional, SequenceConstruct
        "OnnxBackendNodeModelTest.test_sequence_map_add_1_sequence_1_tensor_expanded_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_add_2_sequences_expanded_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_extract_shapes_expanded_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_identity_1_sequence_1_tensor_expanded_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_identity_1_sequence_expanded_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_identity_2_sequences_expanded_cpu",
        "OnnxBackendNodeModelTest.test_split_to_sequence_1_cpu",
        "OnnxBackendNodeModelTest.test_split_to_sequence_2_cpu",
        "OnnxBackendNodeModelTest.test_split_to_sequence_nokeepdims_cpu",
    ),
    (
        xfail_issue_38701,
        "OnnxBackendSimpleModelTest.test_strnorm_model_monday_casesensintive_nochangecase_cpu",
        "OnnxBackendSimpleModelTest.test_strnorm_model_nostopwords_nochangecase_cpu",
        "OnnxBackendSimpleModelTest.test_strnorm_model_monday_empty_output_cpu",
        "OnnxBackendSimpleModelTest.test_strnorm_model_monday_insensintive_upper_twodim_cpu",
        "OnnxBackendSimpleModelTest.test_strnorm_model_monday_casesensintive_lower_cpu",
        "OnnxBackendSimpleModelTest.test_strnorm_model_monday_casesensintive_upper_cpu",
        "OnnxBackendNodeModelTest.test_strnormalizer_nostopwords_nochangecase_cpu",
        "OnnxBackendNodeModelTest.test_strnormalizer_export_monday_casesensintive_nochangecase_cpu",
        "OnnxBackendNodeModelTest.test_strnormalizer_export_monday_insensintive_upper_twodim_cpu",
        "OnnxBackendNodeModelTest.test_strnormalizer_export_monday_casesensintive_lower_cpu",
        "OnnxBackendNodeModelTest.test_strnormalizer_export_monday_empty_output_cpu",
        "OnnxBackendNodeModelTest.test_strnormalizer_export_monday_casesensintive_upper_cpu",
        "OnnxBackendNodeModelTest.test_cast_STRING_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT_to_STRING_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_STRING_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_STRING_expanded_cpu",
        "OnnxBackendNodeModelTest.test_castlike_STRING_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_castlike_STRING_to_FLOAT_expanded_cpu",
        "OnnxBackendNodeModelTest.test_equal_string_broadcast_cpu",
        "OnnxBackendNodeModelTest.test_equal_string_cpu",
    ),
    (
        xfail_issue_33651,
        "OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_batch_onlybigrams_skip5_cpu",
        "OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_onlybigrams_levelempty_cpu",
        "OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_batch_onlybigrams_skip0_cpu",
        "OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_uniandbigrams_skip5_cpu",
        "OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_only_bigrams_skip0_cpu",
        "OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_batch_uniandbigrams_skip5_cpu",
        "OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_onlybigrams_skip5_cpu",
    ),
    (
        xfail_issue_38706,
        "OnnxBackendNodeModelTest.test_split_zero_size_splits_cpu",
    ),
    (
        xfail_issue_33581,
        "OnnxBackendNodeModelTest.test_gather_elements_negative_indices_cpu",
    ),
    (
        xfail_issue_38713,
        "OnnxBackendNodeModelTest.test_momentum_cpu",
        "OnnxBackendNodeModelTest.test_nesterov_momentum_cpu",
        "OnnxBackendNodeModelTest.test_momentum_multiple_cpu",
    ),
    (
        xfail_issue_73538,
        "OnnxBackendNodeModelTest.test_onehot_negative_indices_cpu",
    ),
    (
        xfail_issue_33488,
        "OnnxBackendNodeModelTest.test_maxunpool_export_with_output_shape_cpu",
        "OnnxBackendNodeModelTest.test_maxunpool_export_without_output_shape_cpu",
    ),
    (xfail_issue_38724, "OnnxBackendNodeModelTest.test_resize_tf_crop_and_resize_cpu"),
    (
        xfail_issue_33606,
        "OnnxBackendNodeModelTest.test_det_2d_cpu",
        "OnnxBackendNodeModelTest.test_det_nd_cpu",
    ),
    (
        xfail_issue_38734,
        "OnnxBackendNodeModelTest.test_adam_multiple_cpu",
        "OnnxBackendNodeModelTest.test_adam_cpu",
    ),
    (
        xfail_issue_38735,
        "OnnxBackendNodeModelTest.test_adagrad_multiple_cpu",
        "OnnxBackendNodeModelTest.test_adagrad_cpu",
    ),
    (
        xfail_issue_48052,
        "OnnxBackendNodeModelTest.test_training_dropout_cpu",
        "OnnxBackendNodeModelTest.test_training_dropout_mask_cpu",
        "OnnxBackendNodeModelTest.test_training_dropout_default_cpu",
        "OnnxBackendNodeModelTest.test_training_dropout_zero_ratio_cpu",
        "OnnxBackendNodeModelTest.test_training_dropout_default_mask_cpu",
        "OnnxBackendNodeModelTest.test_training_dropout_zero_ratio_mask_cpu",
    ),
    (
        xfail_issue_45180,
        "OnnxBackendNodeModelTest.test_reduce_sum_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_sum_do_not_keepdims_random_cpu",
    ),
    (
        xfail_issue_44858,
        "OnnxBackendNodeModelTest.test_unsqueeze_axis_0_cpu",
        "OnnxBackendNodeModelTest.test_unsqueeze_axis_1_cpu",
        "OnnxBackendNodeModelTest.test_unsqueeze_axis_2_cpu",
        "OnnxBackendNodeModelTest.test_unsqueeze_negative_axes_cpu",
        "OnnxBackendNodeModelTest.test_unsqueeze_three_axes_cpu",
        "OnnxBackendNodeModelTest.test_unsqueeze_two_axes_cpu",
        "OnnxBackendNodeModelTest.test_unsqueeze_unsorted_axes_cpu",
    ),
    (
        xfail_issue_44965,
        "OnnxBackendNodeModelTest.test_loop13_seq_cpu",
        "OnnxBackendNodeModelTest.test_sequence_insert_at_back_cpu",
        "OnnxBackendNodeModelTest.test_sequence_insert_at_front_cpu",
    ),
    (xfail_issue_58033, "OnnxBackendNodeModelTest.test_einsum_batch_diagonal_cpu"),
    (
        xfail_issue_63033,
        "OnnxBackendNodeModelTest.test_batchnorm_epsilon_training_mode_cpu",
        "OnnxBackendNodeModelTest.test_batchnorm_example_training_mode_cpu",
    ),
    (xfail_issue_63036, "OnnxBackendNodeModelTest.test_convtranspose_autopad_same_cpu"),
    (
        xfail_issue_63043,
        "OnnxBackendNodeModelTest.test_gru_batchwise_cpu",
        "OnnxBackendNodeModelTest.test_lstm_batchwise_cpu",
        "OnnxBackendNodeModelTest.test_simple_rnn_batchwise_cpu",
    ),
    (
        xfail_issue_38710,
        "OnnxBackendNodeModelTest.test_reshape_allowzero_reordered_cpu",
    ),
    (
        skip_dynamic_model,
        "OnnxBackendNodeModelTest.test_triu_one_row_cpu",
        "OnnxBackendNodeModelTest.test_squeeze_cpu",
        "OnnxBackendNodeModelTest.test_squeeze_negative_axes_cpu",
        "OnnxBackendNodeModelTest.test_center_crop_pad_crop_negative_axes_hwc_expanded_cpu",
        "OnnxBackendNodeModelTest.test_constant_pad_negative_axes_cpu",
    ),
    (
        skip_rng_tests,
        "OnnxBackendNodeModelTest.test_bernoulli_cpu",
        "OnnxBackendNodeModelTest.test_bernoulli_double_cpu",
        "OnnxBackendNodeModelTest.test_bernoulli_double_expanded_cpu",
        "OnnxBackendNodeModelTest.test_bernoulli_expanded_cpu",
        "OnnxBackendNodeModelTest.test_bernoulli_seed_cpu",
        "OnnxBackendNodeModelTest.test_bernoulli_seed_expanded_cpu",
    ),
    (
        xfail_issue_63137,
        "OnnxBackendNodeModelTest.test_optional_get_element_cpu",
        "OnnxBackendNodeModelTest.test_optional_get_element_sequence_cpu",
        "OnnxBackendNodeModelTest.test_optional_has_element_cpu",
        "OnnxBackendNodeModelTest.test_optional_has_element_empty_cpu",
        "OnnxBackendNodeModelTest.test_loop16_seq_none_cpu",  # OptionalHasElement, SequenceInsert
    ),
    (
        xfail_issue_63138,
        "OnnxBackendNodeModelTest.test_shape_end_1_cpu",
        "OnnxBackendNodeModelTest.test_shape_end_negative_1_cpu",
        "OnnxBackendNodeModelTest.test_shape_start_1_cpu",
        "OnnxBackendNodeModelTest.test_shape_start_1_end_2_cpu",
        "OnnxBackendNodeModelTest.test_shape_start_1_end_negative_1_cpu",
        "OnnxBackendNodeModelTest.test_shape_start_negative_1_cpu",
    ),
    (
        xfail_issue_69444,
        "OnnxBackendNodeModelTest.test_resize_downsample_scales_linear_align_corners_cpu",
        "OnnxBackendNodeModelTest.test_resize_downsample_scales_cubic_align_corners_cpu",
        "OnnxBackendNodeModelTest.test_resize_downsample_scales_cubic_A_n0p5_exclude_outside_cpu",
        "OnnxBackendNodeModelTest.test_resize_upsample_scales_cubic_A_n0p5_exclude_outside_cpu",
    ),
    (
        skip_segfault,
        "OnnxBackendNodeModelTest.test_sce_NCd1d2d3d4d5_mean_weight_cpu",  # ticket: 81976
        "OnnxBackendNodeModelTest.test_sce_NCd1d2d3d4d5_mean_weight_log_prob_cpu",  # ticket: 81976
        "OnnxBackendNodeModelTest.test_sce_NCd1d2d3d4d5_none_no_weight_cpu",  # ticket: 81976
        "OnnxBackendNodeModelTest.test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_cpu",  # ticket: 81976
        "OnnxBackendNodeModelTest.test_layer_normalization_2d_axis0_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_2d_axis1_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_2d_axis_negative_1_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_2d_axis_negative_2_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis0_epsilon_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis1_epsilon_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis2_epsilon_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis_negative_1_epsilon_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis_negative_2_epsilon_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis_negative_3_epsilon_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis0_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis1_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis2_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis3_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis_negative_1_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis_negative_2_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis_negative_3_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis_negative_4_cpu",  # ticket: 90649
        "OnnxBackendNodeModelTest.test_layer_normalization_default_axis_cpu",  # ticket: 90649
    ),
    (
        xfail_issue_81976,  # SoftmaxCrossEntropyLoss operator
        "OnnxBackendNodeModelTest.test_sce_mean_3d_cpu",
        "OnnxBackendNodeModelTest.test_sce_mean_3d_log_prob_cpu",
    ),
    (
        xfail_issue_82038,
        "OnnxBackendNodeModelTest.test_scatternd_add_cpu",
        "OnnxBackendNodeModelTest.test_scatternd_multiply_cpu",
    ),
    (
        xfail_issue_82039,
        "OnnxBackendNodeModelTest.test_identity_opt_cpu",
    ),
    (
        xfail_issue_90649,
        "OnnxBackendNodeModelTest.test_melweightmatrix_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_add_1_sequence_1_tensor_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_add_2_sequences_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_extract_shapes_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_identity_1_sequence_1_tensor_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_identity_1_sequence_cpu",
        "OnnxBackendNodeModelTest.test_sequence_map_identity_2_sequences_cpu",
        "OnnxBackendNodeModelTest.test_stft_cpu",
        "OnnxBackendNodeModelTest.test_stft_with_window_cpu",
    ),
    (
        xfail_issue_91151,
        "OnnxBackendNodeModelTest.test_castlike_BFLOAT16_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_BFLOAT16_cpu",
    ),
    (
        xfail_issue_99949,
        "OnnxBackendNodeModelTest.test_bitwise_and_i16_3d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_and_i32_2d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_and_ui64_bcast_3v1d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_not_2d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_not_3d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_not_4d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_or_i16_4d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_or_i32_2d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_or_ui64_bcast_3v1d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_xor_ui8_bcast_4v3d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_xor_i16_3d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_xor_i32_2d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_and_ui8_bcast_4v3d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_or_ui8_bcast_4v3d_cpu",
        "OnnxBackendNodeModelTest.test_bitwise_xor_ui64_bcast_3v1d_cpu",
    ),
    (
        xfail_issue_99950,
        "OnnxBackendNodeModelTest.test_center_crop_pad_crop_and_pad_cpu",
        "OnnxBackendNodeModelTest.test_center_crop_pad_crop_axes_chw_cpu",
        "OnnxBackendNodeModelTest.test_center_crop_pad_crop_axes_chw_expanded_cpu",
        "OnnxBackendNodeModelTest.test_center_crop_pad_crop_axes_hwc_cpu",
        "OnnxBackendNodeModelTest.test_center_crop_pad_crop_axes_hwc_expanded_cpu",
        "OnnxBackendNodeModelTest.test_center_crop_pad_crop_cpu",
        "OnnxBackendNodeModelTest.test_center_crop_pad_pad_cpu",
        "OnnxBackendNodeModelTest.test_center_crop_pad_crop_negative_axes_hwc_cpu",
        "OnnxBackendNodeModelTest.test_center_crop_pad_crop_negative_axes_hwc_expanded_cpu",
    ),
    (
        xfail_issue_99952,
        "OnnxBackendNodeModelTest.test_col2im_5d_cpu",
        "OnnxBackendNodeModelTest.test_col2im_cpu",
        "OnnxBackendNodeModelTest.test_col2im_dilations_cpu",
        "OnnxBackendNodeModelTest.test_col2im_pads_cpu",
        "OnnxBackendNodeModelTest.test_col2im_strides_cpu",
    ),
    (
        xfail_issue_99954,
        "OnnxBackendNodeModelTest.test_constant_pad_axes_cpu",
    ),
    (
        xfail_issue_99955,
        "OnnxBackendNodeModelTest.test_group_normalization_epsilon_expanded_cpu",
        "OnnxBackendNodeModelTest.test_group_normalization_example_expanded_cpu",
    ),
    (
        xfail_issue_99957,
        "OnnxBackendNodeModelTest.test_layer_normalization_2d_axis1_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_2d_axis_negative_1_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis1_epsilon_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis2_epsilon_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis_negative_1_epsilon_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_3d_axis_negative_2_epsilon_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis1_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis2_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis3_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis_negative_1_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis_negative_2_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_4d_axis_negative_3_expanded_ver18_cpu",
        "OnnxBackendNodeModelTest.test_layer_normalization_default_axis_expanded_ver18_cpu",
    ),
    (
        xfail_issue_99958,
        "OnnxBackendNodeModelTest.test_logsoftmax_large_number_expanded_ver18_cpu",
    ),
    (
        xfail_issue_99959,
        "OnnxBackendNodeModelTest.test_mish_cpu",
    ),
    (
        xfail_issue_99960,
        "OnnxBackendNodeModelTest.test_mvn_expanded_ver18_cpu",
    ),
    (
        xfail_issue_99961,
        "OnnxBackendNodeModelTest.test_optional_get_element_optional_sequence_cpu",
        "OnnxBackendNodeModelTest.test_optional_get_element_optional_tensor_cpu",
        "OnnxBackendNodeModelTest.test_optional_get_element_tensor_cpu",
        "OnnxBackendNodeModelTest.test_optional_has_element_empty_no_input_name_optional_input_cpu",
        "OnnxBackendNodeModelTest.test_optional_has_element_empty_no_input_name_tensor_input_cpu",
        "OnnxBackendNodeModelTest.test_optional_has_element_empty_no_input_optional_input_cpu",
        "OnnxBackendNodeModelTest.test_optional_has_element_empty_no_input_tensor_input_cpu",
        "OnnxBackendNodeModelTest.test_optional_has_element_empty_optional_input_cpu",
        "OnnxBackendNodeModelTest.test_optional_has_element_optional_input_cpu",
        "OnnxBackendNodeModelTest.test_optional_has_element_tensor_input_cpu",
    ),
    (
        xfail_issue_99968,
        "OnnxBackendNodeModelTest.test_reduce_l1_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l1_do_not_keepdims_example_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l1_do_not_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l1_do_not_keepdims_random_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l1_keep_dims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l1_keep_dims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l1_negative_axes_keep_dims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l1_negative_axes_keep_dims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l2_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l2_do_not_keepdims_example_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l2_do_not_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l2_do_not_keepdims_random_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l2_keep_dims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l2_keep_dims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l2_negative_axes_keep_dims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l2_negative_axes_keep_dims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_asc_axes_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_asc_axes_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_desc_axes_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_desc_axes_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_exp_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_exp_do_not_keepdims_example_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_exp_do_not_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_exp_do_not_keepdims_random_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_l1_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_exp_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_exp_negative_axes_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_exp_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_negative_axes_cpu",
        "OnnxBackendNodeModelTest.test_reduce_max_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_max_do_not_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_max_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_max_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_max_negative_axes_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_max_negative_axes_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_mean_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_log_sum_exp_negative_axes_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_mean_do_not_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_mean_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_mean_negative_axes_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_mean_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_mean_negative_axes_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_min_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_min_do_not_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_min_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_min_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_min_negative_axes_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_min_negative_axes_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_prod_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_prod_do_not_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_prod_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_prod_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_prod_negative_axes_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_prod_negative_axes_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_sum_square_do_not_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_sum_square_do_not_keepdims_example_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_sum_square_do_not_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_sum_square_do_not_keepdims_random_expanded_cpu",
        "OnnxBackendNodeModelTest.test_reduce_sum_square_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_sum_square_keepdims_random_cpu",
        "OnnxBackendNodeModelTest.test_reduce_sum_square_negative_axes_keepdims_example_cpu",
        "OnnxBackendNodeModelTest.test_reduce_sum_square_negative_axes_keepdims_random_cpu",
    ),
    (
        xfail_issue_99969,
        "OnnxBackendNodeModelTest.test_resize_downsample_scales_cubic_antialias_cpu",
        "OnnxBackendNodeModelTest.test_resize_downsample_scales_linear_antialias_cpu",
        "OnnxBackendNodeModelTest.test_resize_downsample_sizes_cubic_antialias_cpu",
        "OnnxBackendNodeModelTest.test_resize_downsample_sizes_linear_antialias_cpu",
        "OnnxBackendNodeModelTest.test_resize_downsample_sizes_nearest_not_smaller_cpu",
        "OnnxBackendNodeModelTest.test_resize_downsample_sizes_nearest_not_larger_cpu",
        "OnnxBackendNodeModelTest.test_resize_tf_crop_and_resize_axes_2_3_cpu",
        "OnnxBackendNodeModelTest.test_resize_tf_crop_and_resize_axes_3_2_cpu",
        "OnnxBackendNodeModelTest.test_resize_upsample_scales_nearest_axes_2_3_cpu",
        "OnnxBackendNodeModelTest.test_resize_upsample_scales_nearest_axes_3_2_cpu",
        "OnnxBackendNodeModelTest.test_resize_upsample_sizes_nearest_axes_2_3_cpu",
        "OnnxBackendNodeModelTest.test_resize_upsample_sizes_nearest_axes_3_2_cpu",
        "OnnxBackendNodeModelTest.test_resize_upsample_sizes_nearest_not_larger_cpu",
    ),
    (
        xfail_issue_99970,
        "OnnxBackendNodeModelTest.test_scatternd_max_cpu",
        "OnnxBackendNodeModelTest.test_scatternd_min_cpu",
    ),
    (
        xfail_issue_99972,
        "OnnxBackendNodeModelTest.test_softmax_large_number_expanded_ver18_cpu",
    ),
    (
        xfail_issue_99973,
        "OnnxBackendNodeModelTest.test_split_1d_uneven_split_opset18_cpu",
        "OnnxBackendNodeModelTest.test_split_2d_uneven_split_opset18_cpu",
    ),
    (
        xfail_issue_101965,
        "OnnxBackendNodeModelTest.test_dft_axis_cpu",
        "OnnxBackendNodeModelTest.test_dft_cpu",
        "OnnxBackendNodeModelTest.test_dft_inverse_cpu",
    ),
    (
        xfail_issue_113506,
        "OnnxBackendNodeModelTest.test_lstm_with_peepholes_cpu",
    ),
    (
        xfail_issue_119896,
        "OnnxBackendNodeModelTest.test_cast_FLOAT16_to_FLOAT8E4M3FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT16_to_FLOAT8E4M3FN_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT16_to_FLOAT8E5M2FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT16_to_FLOAT8E5M2_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT8E4M3FNUZ_to_FLOAT16_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT8E4M3FNUZ_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT8E4M3FN_to_FLOAT16_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT8E4M3FN_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT8E5M2FNUZ_to_FLOAT16_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT8E5M2FNUZ_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT8E5M2_to_FLOAT16_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT8E5M2_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT_to_FLOAT8E4M3FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT_to_FLOAT8E4M3FN_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT_to_FLOAT8E5M2FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_cast_FLOAT_to_FLOAT8E5M2_cpu",
        "OnnxBackendNodeModelTest.test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN_cpu",
        "OnnxBackendNodeModelTest.test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2_cpu",
        "OnnxBackendNodeModelTest.test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN_cpu",
        "OnnxBackendNodeModelTest.test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_cast_no_saturate_FLOAT_to_FLOAT8E5M2_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT8E4M3FN_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT8E5M2_to_FLOAT_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT8E5M2_to_FLOAT_expanded_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_FLOAT8E4M3FN_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_FLOAT8E5M2_cpu",
        "OnnxBackendNodeModelTest.test_castlike_FLOAT_to_FLOAT8E5M2_expanded_cpu",
        "OnnxBackendNodeModelTest.test_dequantizelinear_e4m3fn_cpu",
        "OnnxBackendNodeModelTest.test_dequantizelinear_e5m2_cpu",
        "OnnxBackendNodeModelTest.test_quantizelinear_e4m3fn_cpu",
        "OnnxBackendNodeModelTest.test_quantizelinear_e5m2_cpu",
    ),
    (
        xfail_issue_119900,
        "OnnxBackendNodeModelTest.test_resize_downsample_scales_linear_half_pixel_symmetric_cpu",
        "OnnxBackendNodeModelTest.test_resize_upsample_scales_linear_half_pixel_symmetric_cpu",
    ),
    (
        xfail_issue_119903,
        "OnnxBackendNodeModelTest.test_basic_deform_conv_with_padding_cpu",
        "OnnxBackendNodeModelTest.test_basic_deform_conv_without_padding_cpu",
        "OnnxBackendNodeModelTest.test_deform_conv_with_mask_bias_cpu",
        "OnnxBackendNodeModelTest.test_deform_conv_with_multiple_offset_groups_cpu",
    ),
    (
        xfail_issue_119906,
        "OnnxBackendNodeModelTest.test_lppool_1d_default_cpu",
        "OnnxBackendNodeModelTest.test_lppool_2d_default_cpu",
        "OnnxBackendNodeModelTest.test_lppool_2d_dilations_cpu",
        "OnnxBackendNodeModelTest.test_lppool_2d_pads_cpu",
        "OnnxBackendNodeModelTest.test_lppool_2d_same_lower_cpu",
        "OnnxBackendNodeModelTest.test_lppool_2d_same_upper_cpu",
        "OnnxBackendNodeModelTest.test_lppool_2d_strides_cpu",
        "OnnxBackendNodeModelTest.test_lppool_3d_default_cpu",
    ),
    (
        xfail_issue_119919,
        "OnnxBackendNodeModelTest.test_wrap_pad_cpu",
    ),
    (
        xfail_issue_119922,
        "OnnxBackendNodeModelTest.test_ai_onnx_ml_array_feature_extractor_cpu",
        "OnnxBackendNodeModelTest.test_ai_onnx_ml_binarizer_cpu",
    ),
    (
        xfail_issue_119925,
        "OnnxBackendNodeModelTest.test_averagepool_2d_dilations_cpu",
    ),
    (
        xfail_issue_119926,
        "OnnxBackendNodeModelTest.test_roialign_mode_max_cpu",
    ),
]

if platform.system() == 'Darwin':
    tests_expected_to_fail.extend([
        (
            skip_issue_58676,
            "OnnxBackendNodeModelTest.test_mish_expanded_cpu"
        ),
        (
            skip_issue_58676,
            "OnnxBackendNodeModelTest.test_resize_downsample_scales_linear_cpu"
        ),
        (
            skip_issue_58676,
            "OnnxBackendNodeModelTest.test_div_uint8_cpu"
        )]
    )

for test_group in tests_expected_to_fail:
    for test_case in test_group[1:]:
        expect_fail(f"{test_case}", test_group[0])
