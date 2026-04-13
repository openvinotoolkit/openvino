// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/atan2_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/atan2.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace testing;

namespace {

void run_atan2_decomposition_test(element::Type elem_type, const Shape& shape_y, const Shape& shape_x) {
    auto y = std::make_shared<op::v0::Parameter>(elem_type, shape_y);
    auto x = std::make_shared<op::v0::Parameter>(elem_type, shape_x);
    auto atan2 = std::make_shared<op::v17::Atan2>(y, x);
    auto model = std::make_shared<Model>(OutputVector{atan2}, ParameterVector{y, x});

    pass::Manager manager;
    manager.register_pass<pass::Atan2Decomposition>();
    manager.run_passes(model);

    // Verify no Atan2 nodes remain after decomposition.
    for (const auto& node : model->get_ordered_ops()) {
        ASSERT_FALSE(ov::is_type<op::v17::Atan2>(node))
            << "Atan2 node should have been decomposed, but found: " << node->get_friendly_name();
    }

    // The final output should be a Select node.
    auto result_node = model->get_result()->get_input_node_shared_ptr(0);
    ASSERT_TRUE(ov::is_type<op::v1::Select>(result_node))
        << "Expected Select as output, got: " << result_node->get_type_info().name;
}

}  // namespace

TEST(Atan2DecompositionTest, BasicF32) {
    run_atan2_decomposition_test(element::f32, Shape{3, 2}, Shape{3, 2});
}

TEST(Atan2DecompositionTest, FP16) {
    run_atan2_decomposition_test(element::f16, Shape{2, 4}, Shape{2, 4});
}

TEST(Atan2DecompositionTest, Broadcast) {
    run_atan2_decomposition_test(element::f32, Shape{3, 1}, Shape{1, 4});
}

TEST(Atan2DecompositionTest, BF16) {
    run_atan2_decomposition_test(element::bf16, Shape{4}, Shape{4});
}

TEST(Atan2DecompositionTest, F64) {
    run_atan2_decomposition_test(element::f64, Shape{2, 3, 4}, Shape{2, 3, 4});
}
