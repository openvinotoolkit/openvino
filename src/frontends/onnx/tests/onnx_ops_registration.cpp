// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>

#include <algorithm>
#include <sstream>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
/*
OPENVINO_TEST(ops_registration, check_importing_abs_in_all_opset_versions) {
    ONNXModelEditor editor{
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "abs.onnx"})};
    for (int version = 1; version <= ONNX_OPSET_VERSION; ++version) {
        const auto changed_opset_model = change_opset_version(editor.model_string(), {version});
        std::stringstream model_stream{changed_opset_model};
        OV_ASSERT_NO_THROW(ONNXModelEditor(model_stream).get_function());
    }
}

OPENVINO_TEST(ops_registration, check_importing_add_in_different_opsets) {
    const auto legacy_broadcast_detected = [](const std::vector<std::shared_ptr<ov::Node>>& ops) {
        return std::find_if(std::begin(ops), std::end(ops), [](const std::shared_ptr<ov::Node>& op) {
                   return std::string{op->get_type_name()} == "Reshape";
               }) != std::end(ops);
    };
    ONNXModelEditor editor{ov::util::path_join(
        {ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "add_v6_broadcast_dynamic.onnx"})};
    for (int version = 1; version <= ONNX_OPSET_VERSION; ++version) {
        const auto changed_opset_model = change_opset_version(editor.model_string(), {version});
        std::stringstream model_stream{changed_opset_model};
        const auto model = ONNXModelEditor(model_stream).get_function();
        const bool legacy_add = legacy_broadcast_detected(model->get_ops());
        if (version <= 6) {
            ASSERT_TRUE(legacy_add);
        } else {
            ASSERT_FALSE(legacy_add);
        }
    }
}
*/
