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

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

// is there any benefit of running below tests on different backends?
// why are these here anyway?

OPENVINO_TEST(${BACKEND_NAME}, add_abc_from_ir) {
    const auto model = read_ir("core/models/ir/add_abc.xml");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, add_abc_from_ir_with_bin_path) {
    const auto model = read_ir("core/models/ir/add_abc.xml", "core/models/ir/add_abc.bin");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});

    test_case.run();
}
