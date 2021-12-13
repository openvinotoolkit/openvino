// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "editor.hpp"
#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/test_control.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
OPENVINO_SUPPRESS_DEPRECATED_START

// ############################################################################ CORE TESTS
NGRAPH_TEST(TEMPLATE, onnx_compress_axis_0) {
    ov::onnx_editor::ONNXModelEditor editor{file_util::path_join(SERIALIZED_ZOO, "onnx/compress_0.onnx")};

    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("input", op::Constant::create(element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::Constant::create(element::boolean, Shape{3}, {false, true, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase(function);

    test_case.add_expected_output<float>(Shape{2, 2}, {3., 4., 5., 6.});
    test_case.run();
}

NGRAPH_TEST(TEMPLATE, onnx_compress_axis_1) {
    ov::onnx_editor::ONNXModelEditor editor{file_util::path_join(SERIALIZED_ZOO, "onnx/compress_1.onnx")};

    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("input", op::Constant::create(element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::Constant::create(element::boolean, Shape{2}, {false, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase(function);

    test_case.add_expected_output<float>(Shape{3, 1}, {2., 4., 6.});
    test_case.run();
}

NGRAPH_TEST(TEMPLATE, onnx_compress_default_axis) {
    ov::onnx_editor::ONNXModelEditor editor{file_util::path_join(SERIALIZED_ZOO, "onnx/compress_default_axis.onnx")};

    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("input", op::Constant::create(element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::Constant::create(element::boolean, Shape{5}, {false, true, false, false, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase(function);

    test_case.add_expected_output<float>(Shape{2}, {2., 5.});
    test_case.run();
}

NGRAPH_TEST(TEMPLATE, onnx_compress_negative_axis) {
    ov::onnx_editor::ONNXModelEditor editor{file_util::path_join(SERIALIZED_ZOO, "onnx/compress_negative_axis.onnx")};

    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("input", op::Constant::create(element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::Constant::create(element::boolean, Shape{2}, {false, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase(function);

    test_case.add_expected_output<float>(Shape{3, 1}, {2., 4., 6.});
    test_case.run();
}
