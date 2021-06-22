// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parser_tests.hpp"

ir_test_params ir_test_cases[] = {
    ir_test_params("MYRIAD", "FP16", negative_conv_kernel_x_case),
    ir_test_params("MYRIAD", "FP16", negative_conv_kernel_y_case),
    ir_test_params("MYRIAD", "FP16", negative_conv_stride_x_case),
    ir_test_params("MYRIAD", "FP16", negative_conv_weights_case),
    ir_test_params("MYRIAD", "FP16", negative_conv_biases_case),

    ir_test_params("MYRIAD", "FP16", negative_fc_out_size_case),
    ir_test_params("MYRIAD", "FP16", negative_fc_weights_case),
    ir_test_params("MYRIAD", "FP16", negative_fc_biases_case),

    ir_test_params("MYRIAD", "FP16", negative_deconv_kernel_x_case),
    ir_test_params("MYRIAD", "FP16", negative_deconv_kernel_y_case),
    ir_test_params("MYRIAD", "FP16", negative_deconv_stride_x_case),
    ir_test_params("MYRIAD", "FP16", negative_deconv_weights_case),
    ir_test_params("MYRIAD", "FP16", negative_deconv_biases_case),

    ir_test_params("MYRIAD", "FP16", negative_pool_kernel_x_case),
    ir_test_params("MYRIAD", "FP16", negative_pool_kernel_y_case),
    ir_test_params("MYRIAD", "FP16", negative_pool_stride_x_case),
    ir_test_params("MYRIAD", "FP16", incorrect_pool_type_case),

    ir_test_params("MYRIAD", "FP16", negative_norm_local_size_case),
    ir_test_params("MYRIAD", "FP16", negative_norm_k_case)
};

INSTANTIATE_TEST_SUITE_P(FunctionalTest_nightly, IncorrectIRTests,
        ::testing::ValuesIn(ir_test_cases),
        getTestName);
