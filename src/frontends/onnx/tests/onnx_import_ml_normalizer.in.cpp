// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#    define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#    define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include <cmath>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::test;
using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

// -----------------------------------------------------------------------
// ai.onnx.ml.Normalizer – norm="L2"
//
// Input X shape: [3, 4]
// Each row is divided by its L2 norm (sqrt of sum of squares).
// -----------------------------------------------------------------------
OPENVINO_TEST(${BACKEND_NAME}, onnx_ai_onnx_ml_normalizer_l2) {
    const auto model = convert_model("ai.onnx.ml/normalizer_l2.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off
    test_case.add_input<float>({
        1.0f, 2.0f,  3.0f,  4.0f,
        5.0f, 6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    });
    // clang-format on

    // Expected: divide each row by its L2 norm
    const float n0 = std::sqrt(1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f + 4.0f * 4.0f);        // sqrt(30)
    const float n1 = std::sqrt(5.0f * 5.0f + 6.0f * 6.0f + 7.0f * 7.0f + 8.0f * 8.0f);        // sqrt(174)
    const float n2 = std::sqrt(9.0f * 9.0f + 10.0f * 10.0f + 11.0f * 11.0f + 12.0f * 12.0f);  // sqrt(446)

    // clang-format off
    test_case.add_expected_output<float>(
        ov::Shape{3, 4},
        {
            1.0f / n0,  2.0f / n0,  3.0f / n0,  4.0f / n0,
            5.0f / n1,  6.0f / n1,  7.0f / n1,  8.0f / n1,
            9.0f / n2, 10.0f / n2, 11.0f / n2, 12.0f / n2
        });
    // clang-format on

    test_case.run();
}

// -----------------------------------------------------------------------
// ai.onnx.ml.Normalizer – norm="L1"
//
// Input X shape: [2, 3]
// Each row is divided by its L1 norm (sum of absolute values).
// -----------------------------------------------------------------------
OPENVINO_TEST(${BACKEND_NAME}, onnx_ai_onnx_ml_normalizer_l1) {
    const auto model = convert_model("ai.onnx.ml/normalizer_l1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off
    test_case.add_input<float>({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });
    // clang-format on

    // Expected: divide each row by its L1 norm (sum of abs values)
    const float n0 = 1.0f + 2.0f + 3.0f;  // 6.0
    const float n1 = 4.0f + 5.0f + 6.0f;  // 15.0

    // clang-format off
    test_case.add_expected_output<float>(
        ov::Shape{2, 3},
        {
            1.0f / n0, 2.0f / n0, 3.0f / n0,
            4.0f / n1, 5.0f / n1, 6.0f / n1
        });
    // clang-format on

    test_case.run();
}

// -----------------------------------------------------------------------
// ai.onnx.ml.Normalizer – norm="MAX"
//
// Input X shape: [2, 5]
// Each row is divided by its MAX value (maximum of absolute values).
// -----------------------------------------------------------------------
OPENVINO_TEST(${BACKEND_NAME}, onnx_ai_onnx_ml_normalizer_max) {
    const auto model = convert_model("ai.onnx.ml/normalizer_max.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off
    test_case.add_input<float>({
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 2.0f, 8.0f, 1.0f, 4.0f
    });
    // clang-format on

    // Expected: divide each row by its max element
    const float m0 = 5.0f;  // max of row 0
    const float m1 = 8.0f;  // max of row 1

    // clang-format off
    test_case.add_expected_output<float>(
        ov::Shape{2, 5},
        {
            1.0f / m0, 2.0f / m0, 3.0f / m0, 4.0f / m0, 5.0f / m0,
            6.0f / m1, 2.0f / m1, 8.0f / m1, 1.0f / m1, 4.0f / m1
        });
    // clang-format on

    test_case.run();
}
