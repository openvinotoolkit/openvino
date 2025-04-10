// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "common_test_utils/all_close.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");

namespace {
template <typename T>
void test_constant_folding(std::shared_ptr<ov::Model> ov_model,
                           const std::vector<T> expected_output,
                           const PartialShape expected_shape = PartialShape::dynamic()) {
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(ov_model);

    for (auto ng_node : ov_model->get_ordered_ops()) {
        if (ov::is_type<op::v0::Constant>(ng_node)) {
            const auto folded_node = ov::as_type_ptr<op::v0::Constant>(ng_node);
            const auto output_values = folded_node->cast_vector<T>();

            EXPECT_TRUE(ov::test::utils::all_close(expected_output, output_values));

            if (expected_shape.is_static()) {
                EXPECT_EQ(folded_node->get_output_shape(0), expected_shape.to_shape());
            }

            return;
        }
    }

    FAIL() << "ONNX model import with constant folding failed.";
}
}  // namespace

OPENVINO_TEST(${BACKEND_NAME}, onnx_const_folding_model_scatter_elements) {
    const auto fn = convert_model("scatter_elements_opset11.onnx");

    test_constant_folding<float>(fn, {1.0f, 1.1f, 3.0f, 2.1f, 5.0f}, Shape{1, 5});
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_scalar) {
    const auto fn = convert_model("non_zero_scalar.onnx");

    test_constant_folding<int64_t>(fn, {0}, Shape{1, 1});
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_1d) {
    const auto fn = convert_model("non_zero_1d.onnx");

    test_constant_folding<int64_t>(fn, {1, 2, 4}, Shape{1, 3});
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_1d_float) {
    const auto fn = convert_model("non_zero_1d_float.onnx");

    test_constant_folding<int64_t>(fn, {0, 1, 3, 4, 5, 6, 7, 8, 9});
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_3d) {
    const auto fn = convert_model("non_zero_3d.onnx");

    // Vertical slices are 3D indices of non-zero elements in the input tensor
    // {0, 0, 0, 1, 1, 2, 2}
    // {0, 0, 1, 0, 1, 0, 1}
    // {0, 1, 1, 1, 0, 0, 1}
    test_constant_folding<int64_t>(fn, {0, 0, 0, 1, 1, 2, 2, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1});
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_2d_bool) {
    const auto fn = convert_model("non_zero_2d_bool.onnx");

    test_constant_folding<int64_t>(fn, {0, 1, 1, 0});
}
