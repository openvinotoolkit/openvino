// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>

#include <algorithm>
#include <sstream>

#include "common_test_utils/file_utils.hpp"
#include "editor.hpp"
#include "engines_util/test_case.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "onnx_test_util.hpp"
#include "util/test_control.hpp"

using namespace ov;
using namespace ov::onnx_editor;
using namespace ngraph::test;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(ops_registration, check_importing_abs_in_all_opset_versions) {
    ONNXModelEditor editor{
        ngraph::file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/abs.onnx")};
    for (int version = 1; version <= ONNX_OPSET_VERSION; ++version) {
        const auto changed_opset_model = change_opset_version(editor.model_string(), {version});
        std::stringstream model_stream{changed_opset_model};
        ONNXModelEditor(model_stream).get_function();
    }
}
