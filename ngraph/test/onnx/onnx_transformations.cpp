// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>

#include "editor.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "onnx_test_util.hpp"
#include "util/test_control.hpp"

static std::string s_manifest = "${MANIFEST}";

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;
using namespace onnx_editor;
using namespace ngraph::test;

namespace {
// Names of input and output names of nodes after a function expanding have names based on a node address.
// As a result, the names are different during each tests execution.
// It requires custom way of name comparison.
// https://github.com/onnx/onnx/blob/767f752829f83dbc9bd0a364d6138890f667fc38/onnx/defs/function.cc#L23
bool after_func_expand_name_comp(const std::string& lhs, const std::string& rhs) {
    std::regex address_pattern("(0x)?[0-9A-Fa-f]{8,}");

    const std::string lhs_sanitized = std::regex_replace(lhs, address_pattern, "");
    const std::string rhs_sanitized = std::regex_replace(rhs, address_pattern, "");

    return lhs_sanitized == rhs_sanitized;
}
}  // namespace

NGRAPH_TEST(onnx_transformations, expand_function_greater_or_equal) {
    ONNXModelEditor editor{file_util::path_join(SERIALIZED_ZOO, "onnx/transformations/greater_or_equal.onnx")};
    editor.decode();  // onnx transformations are applied

    const auto ref_model = file_util::path_join(SERIALIZED_ZOO,
                                                "onnx/transformations/reference/"
                                                "greater_or_equal_expanded.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model, after_func_expand_name_comp);
    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_transformations, expand_function_softmax_crossentropy) {
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/transformations/softmax_crossentropy_consumed.onnx")};
    editor.decode();  // onnx transformations are applied

    const auto ref_model = file_util::path_join(SERIALIZED_ZOO,
                                                "onnx/transformations/reference/"
                                                "softmax_crossentropy_consumed_expanded.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model, after_func_expand_name_comp);
    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_transformations, expand_function_dynamic_quantize_linear) {
    ONNXModelEditor editor{file_util::path_join(SERIALIZED_ZOO, "onnx/transformations/dynamic_quantize_linear.onnx")};
    editor.decode();  // onnx transformations are applied

    const auto ref_model = file_util::path_join(SERIALIZED_ZOO,
                                                "onnx/transformations/reference/"
                                                "dynamic_quantize_linear_expanded.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model, after_func_expand_name_comp);
    EXPECT_TRUE(result.is_ok) << result.error_message;
}
