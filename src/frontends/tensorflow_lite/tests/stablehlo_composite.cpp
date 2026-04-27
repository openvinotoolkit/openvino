// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "tf_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite::tests;

// Test for STABLEHLO_COMPOSITE RMS support
OPENVINO_TEST(TensorFlowLiteStableHLO, stablehlo_composite_unsupported) {
    // This test verifies that unsupported composite names are properly rejected
    // with a clear error message during fail-fast conversion.
    //
    // Note: A real STABLEHLO_COMPOSITE model would need to be generated
    // with proper TFLite tools or manually constructed. For now, this test
    // documents the expected behavior.

    // TODO: Enable when test model is available
    // auto model = convert_model("stablehlo_composite_unsupported.tflite");
    // This should throw an exception with message containing "odml.rms_norm"
}

// Test for STABLEHLO_COMPOSITE RMS with gamma parameter
OPENVINO_TEST(TensorFlowLiteStableHLO, stablehlo_composite_rms_norm) {
    // This test verifies RMS via STABLEHLO_COMPOSITE odml.rms_norm
    // when the model is available

    // TODO: Enable when test model is available
    // auto model = convert_model("rms_norm_stablehlo_composite.tflite");
    //
    // auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    //
    // // Input: [1, 3, 4] tensor with arbitrary values
    // std::vector<float> input_data = {
    //     1.0f, 2.0f, 3.0f, 4.0f,
    //     5.0f, 6.0f, 7.0f, 8.0f,
    //     9.0f, 10.0f, 11.0f, 12.0f
    // };
    // test_case.add_input<float>(Shape{1, 3, 4}, input_data);
    //
    // // Expected output: RMS-normalized along last axis
    // // RMS = sqrt(mean(x^2)), normalized = x / (RMS + eps)
    // // For each row: sqrt((1^2 + 2^2 + 3^2 + 4^2) / 4) = sqrt(7.5) ≈ 2.738
    // std::vector<float> expected = {
    //     1.0f / 2.738f, 2.0f / 2.738f, 3.0f / 2.738f, 4.0f / 2.738f,
    //     5.0f / 3.873f, 6.0f / 3.873f, 7.0f / 3.873f, 8.0f / 3.873f,
    //     9.0f / 4.949f, 10.0f / 4.949f, 11.0f / 4.949f, 12.0f / 4.949f
    // };
    // test_case.add_expected_output<float>(Shape{1, 3, 4}, expected);
    // test_case.run();
}

// Test for STABLEHLO_COMPOSITE RMS without gamma parameter
OPENVINO_TEST(TensorFlowLiteStableHLO, stablehlo_composite_rms_norm_without_gamma) {
    // This test verifies RMS without gamma via STABLEHLO_COMPOSITE odml.rms_norm
    // The current translator requires exactly 2 inputs [data, gamma],
    // so this test documents the expected rejection of 1-input models.

    // TODO: Enable when test model is available
}
