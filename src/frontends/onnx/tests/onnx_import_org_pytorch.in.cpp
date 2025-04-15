// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_adaptive_avg_pooling2d_nchw) {
    const auto model = convert_model("org.pytorch/adaptive_avg_pooling2d_nchw.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.9945f,
                                0.3466f,
                                0.2894f,
                                0.9318f,
                                0.0115f,
                                0.4867f,
                                0.7608f,
                                0.1550f,
                                0.8485f,
                                0.4971f,
                                0.8833f,
                                0.4579f,
                                0.3673f,
                                0.5410f,
                                0.2004f,
                                0.1519f});
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, {0.4598249f, 0.5342500f, 0.5634750f, 0.4233750f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_adaptive_avg_pooling2d_chw) {
    const auto model = convert_model("org.pytorch/adaptive_avg_pooling2d_chw.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({12.0f, -1.0f, -56.0f, 20.0f, 1.0f, -8.0f, 7.0f, 9.0f});

    test_case.add_expected_output<float>(Shape{1, 2, 2}, {5.5f, -18.0f, -3.5f, 8.0f});
    test_case.run();
}
