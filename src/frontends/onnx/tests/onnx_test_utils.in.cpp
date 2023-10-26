// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <sstream>

#include "common_test_utils/all_close.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "default_opset.hpp"
#include "editor.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_utils.hpp"

using namespace ngraph;
OPENVINO_SUPPRESS_DEPRECATED_START

static std::string s_manifest = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(), "${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

// is there any benefit of running below tests on different backends?
// why are these here anyway?

OPENVINO_TEST(${BACKEND_NAME}, add_abc_from_ir) {
    const auto ir_xml =
        file_util::path_join(ov::test::utils::getExecutableDirectory(), TEST_MODEL_ZOO, "core/models/ir/add_abc.xml");
    const auto function = function_from_ir(ir_xml);

    auto test_case = ov::test::TestCase(function, s_device);
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, add_abc_from_ir_with_bin_path) {
    const auto ir_xml =
        file_util::path_join(ov::test::utils::getExecutableDirectory(), TEST_MODEL_ZOO, "core/models/ir/add_abc.xml");
    const auto ir_bin =
        file_util::path_join(ov::test::utils::getExecutableDirectory(), TEST_MODEL_ZOO, "core/models/ir/add_abc.bin");
    const auto function = function_from_ir(ir_xml, ir_bin);

    auto test_case = ov::test::TestCase(function, s_device);
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});

    test_case.run();
}
