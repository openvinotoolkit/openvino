// Copyright (C) 2018-2022 Intel Corporation
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
#include "default_opset.hpp"
#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "onnx_import/onnx.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
static std::string s_device = test::backend_name_to_device("${BACKEND_NAME}");

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_adaptive_avg_pooling2d_nchw) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/org.pytorch/adaptive_avg_pooling2d_nchw.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0.9945,
                                0.3466,
                                0.2894,
                                0.9318,
                                0.0115,
                                0.4867,
                                0.7608,
                                0.1550,
                                0.8485,
                                0.4971,
                                0.8833,
                                0.4579,
                                0.3673,
                                0.5410,
                                0.2004,
                                0.1519});
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, {0.4598249, 0.5342500, 0.5634750, 0.4233750});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_adaptive_avg_pooling2d_chw) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/org.pytorch/adaptive_avg_pooling2d_chw.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({12.0, -1.0, -56.0, 20.0, 1.0, -8.0, 7.0, 9.0});

    test_case.add_expected_output<float>(Shape{1, 2, 2}, {5.5, -18.0, -3.5, 8.0});
    test_case.run();
}
