// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "onnx_import/onnx.hpp"
#include "util/all_close.hpp"
#include "util/test_control.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;
using namespace ngraph::onnx_import;

static std::string s_manifest = "${MANIFEST}";

namespace {
template <typename T>
void test_constant_folding(std::shared_ptr<ngraph::Function> ng_function,
                           const std::vector<T> expected_output,
                           const PartialShape expected_shape = PartialShape::dynamic()) {
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(ng_function);

    for (auto ng_node : ng_function->get_ordered_ops()) {
        if (op::is_constant(ng_node)) {
            const auto folded_node = ov::as_type_ptr<default_opset::Constant>(ng_node);
            const auto output_values = folded_node->cast_vector<T>();

            EXPECT_TRUE(ngraph::test::all_close(expected_output, output_values));

            if (expected_shape.is_static()) {
                EXPECT_EQ(folded_node->get_output_shape(0), expected_shape.to_shape());
            }

            return;
        }
    }

    FAIL() << "ONNX model import with constant folding failed.";
}
}  // namespace

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_scatter_elements) {
    const auto fn =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_elements_opset11.onnx"));

    test_constant_folding<float>(fn, {1.0, 1.1, 3.0, 2.1, 5.0}, Shape{1, 5});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_scalar) {
    const auto fn = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_scalar.onnx"));

    test_constant_folding<int64_t>(fn, {0}, Shape{1, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_1d) {
    const auto fn = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_1d.onnx"));

    test_constant_folding<int64_t>(fn, {1, 2, 4}, Shape{1, 3});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_1d_float) {
    const auto fn = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_1d_float.onnx"));

    test_constant_folding<int64_t>(fn, {0, 1, 3, 4, 5, 6, 7, 8, 9});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_3d) {
    const auto fn = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_3d.onnx"));

    // Vertical slices are 3D indices of non-zero elements in the input tensor
    // {0, 0, 0, 1, 1, 2, 2}
    // {0, 0, 1, 0, 1, 0, 1}
    // {0, 1, 1, 1, 0, 0, 1}
    test_constant_folding<int64_t>(fn, {0, 0, 0, 1, 1, 2, 2, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_2d_bool) {
    const auto fn = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_2d_bool.onnx"));

    test_constant_folding<int64_t>(fn, {0, 1, 1, 0});
}
