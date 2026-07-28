// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"
#include "conversion_extension.hpp"
#include "gtest/gtest.h"
#include "openvino/op/convert.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "tf_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite::tests;

static std::string s_manifest = "";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_quantize_int8) {
    auto model = convert_model("quantize_int8.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<float>({-40.f, -36.f, -10.30f, -9.15f, -0.85f, -0.5f, 0.f, 0.5f, 0.85f, 1.f, 27.75f, 40.f});
    test_case.add_expected_output<int8_t>(Shape{12}, {-128, -128, -25, -21, 13, 14, 16, 18, 19, 20, 127, 127});
    test_case.run();
}

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_quantize_uint8) {
    auto model = convert_model("quantize_uint8.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<float>({-40.f, -4.25f, -4.f, 0.f, 0.001f, 0.5f, 0.85f, 1.f, 1.22f, 1.27f, 59.75f, 80.f});
    test_case.add_expected_output<uint8_t>(Shape{12}, {0, 0, 0, 16, 16, 18, 19, 20, 21, 21, 255, 255});
    test_case.run();
}

#if (defined OPENVINO_ARCH_ARM && defined(__linux__))
// Ticket: 153164
OPENVINO_TEST(TensorFlowLiteTrickyModels, DISABLED_tflite_dequantize) {
#else
OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_dequantize) {
#endif
    auto model = convert_model("dequantize.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<float>({1, 1, 1, 1});
    test_case.add_expected_output<float>(Shape{2, 2}, {2, 1.75f, 2001, 0.876f});
    test_case.run_with_tolerance_as_fp(0.001f);
}

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_dequantize_int8) {
    auto model = convert_model("dequantize_int8.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<int8_t>({-128, -100, -1, 0, 1, 2, 3, 4, 5, 16, 100, 127});
    test_case.add_expected_output<float>(
        Shape{12},
        {-36.f, -29.f, -4.25f, -4.f, -3.75f, -3.5f, -3.25f, -3.f, -2.75f, 0.f, 21.f, 27.75f});
    test_case.run();
}

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_dequantize_uint8) {
    auto model = convert_model("dequantize_uint8.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<uint8_t>({0, 1, 2, 3, 4, 5, 16, 100, 127, 128, 200, 255});
    test_case.add_expected_output<float>(
        Shape{12},
        {-4.f, -3.75f, -3.5f, -3.25f, -3.f, -2.75f, 0.f, 21.f, 27.75f, 28.f, 46.f, 59.75f});
    test_case.run_with_tolerance_as_fp(0.001f);
}

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_dequantize_fp16) {
    // Tests that a fully fp16 model dequantizes int8 weights to float16 (not float32),
    // avoiding operand type mismatches in ops like FULLY_CONNECTED with fp16 activations.
    auto model = convert_model("dequantize_fp16.tflite");

    // Verify output type is float16
    ASSERT_EQ(model->get_output_element_type(0), ov::element::f16);

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<int8_t>({-128, -100, -1, 0, 1, 2, 3, 4, 5, 16, 100, 127});
    // Expected: (val - zero_point) * scale = (val - 16) * 0.25
    test_case.add_expected_output<ov::float16>(Shape{12},
                                               {ov::float16(-36.f),
                                                ov::float16(-29.f),
                                                ov::float16(-4.25f),
                                                ov::float16(-4.f),
                                                ov::float16(-3.75f),
                                                ov::float16(-3.5f),
                                                ov::float16(-3.25f),
                                                ov::float16(-3.f),
                                                ov::float16(-2.75f),
                                                ov::float16(0.f),
                                                ov::float16(21.f),
                                                ov::float16(27.75f)});
    test_case.run_with_tolerance_as_fp(0.01f);
}

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_quantize_dequantize_int8) {
    auto model = convert_model("qdq_int8.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<float>({-40.f, -36.f, -10.30f, -9.15f, -0.85f, -0.5f, 0.f, 0.5f, 0.85f, 1.f, 27.75f, 40.f});
    test_case.add_expected_output<float>(
        Shape{12},
        {-36.f, -36.f, -10.25f, -9.25, -0.75f, -0.5f, 0.f, 0.5f, 0.75f, 1.f, 27.75f, 27.75f});
    test_case.run_with_tolerance_as_fp(0.001f);
}

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_quantize_dequantize_uint8) {
    auto model = convert_model("qdq_uint8.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<float>({-40.f, -4.25f, -4.f, 0.f, 0.001f, 0.5f, 0.85f, 1.f, 1.22f, 1.27f, 59.75f, 80.f});
    test_case.add_expected_output<float>(Shape{12},
                                         {-4.f, -4.f, -4.f, 0.f, 0.f, 0.5f, 0.75f, 1.f, 1.25f, 1.25f, 59.75f, 59.75f});
    test_case.run_with_tolerance_as_fp(0.001f);
}

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_qdq_convert_is_marked_disable_cf) {
    auto model = convert_model("qdq_uint8.tflite");
    bool has_cf_disabled_convert = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Convert>(op) && ov::pass::constant_folding_is_disabled(op)) {
            has_cf_disabled_convert = true;
            break;
        }
    }
    EXPECT_TRUE(has_cf_disabled_convert);
}
