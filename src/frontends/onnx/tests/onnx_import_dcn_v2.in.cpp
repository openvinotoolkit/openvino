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
#include "editor.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
static std::string s_device = test::backend_name_to_device("${BACKEND_NAME}");

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dcn_v2) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/COP/fairmot.onnx"));
}

// NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dcn_v2_serialize) {
//     // const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
//     //                                                                           SERIALIZED_ZOO,
//     //                                                                           "onnx/COP/fairmot.onnx"));
//     ONNXModelEditor editor{ngraph::file_util::path_join(CommonTestUtils::getExecutableDirectory(),
//                                                             SERIALIZED_ZOO,
//                                                             "onnx/COP/fairmot.onnx")};
//     editor
// }


