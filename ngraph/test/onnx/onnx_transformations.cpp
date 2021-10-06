// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "editor.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"
#include "util/onnx_test_util.hpp"
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
// https://github.com/onnx/onnx/blob/master/onnx/defs/function.cc#L23
bool after_func_expand_name_comp(std::string lhs, std::string rhs) {
    const int lhs_begin_address_pos = lhs.find("0x");
    const int rhs_begin_address_pos = rhs.find("0x");

    if (lhs_begin_address_pos != std::string::npos) {
        lhs.erase(lhs_begin_address_pos, 14);
    }
    if (rhs_begin_address_pos != std::string::npos) {
        rhs.erase(rhs_begin_address_pos, 14);
    }

    return lhs == rhs;
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
