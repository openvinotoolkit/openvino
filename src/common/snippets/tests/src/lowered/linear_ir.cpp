// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/result.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/reshape.hpp"
#include "snippets/op/online_softmax.hpp"
#include "openvino/op/parameter.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/port_descriptor.hpp"

#include "lir_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets;

TEST(LinearIRReplaceWithNode, PreservesPerOutputDescriptors) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 6});
    const auto op = std::make_shared<ov::snippets::op::OnlineSoftmax>(param);
    const auto result_0 = std::make_shared<ov::op::v0::Result>(op->output(0));
    const auto result_1 = std::make_shared<ov::op::v0::Result>(op->output(1));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{result_0, result_1}, ov::ParameterVector{param});

    auto factory = std::make_shared<ov::snippets::IShapeInferSnippetsFactory>();
    ov::snippets::lowered::LinearIR linear_ir(model, factory);

    const auto& online_sm_expr = linear_ir.get_expr_by_node(op);
    ASSERT_NE(nullptr, online_sm_expr);

    const std::vector<VectorDims> subtensors = {get_default_subtensor(2), VectorDims{1, 3}, VectorDims{1, 6}};
    const std::vector<VectorDims> layouts = {VectorDims{}, VectorDims{0, 1}, VectorDims{1, 0}};
    init_expr_descriptors(online_sm_expr, subtensors, layouts);

    const auto expected_desc_0 = online_sm_expr->get_output_port_descriptor(0)->clone();
    const auto expected_desc_1 = online_sm_expr->get_output_port_descriptor(1)->clone();

    ASSERT_NE(expected_desc_0->get_subtensor(), expected_desc_1->get_subtensor());
    ASSERT_NE(expected_desc_0->get_layout(), expected_desc_1->get_layout());

    const auto new_node = std::make_shared<ov::snippets::op::OnlineSoftmax>(param);
    linear_ir.replace_with_node({online_sm_expr}, new_node);

    const auto& new_expr = linear_ir.get_expr_by_node(new_node);
    ASSERT_NE(nullptr, new_expr);

    const auto& new_desc_0 = new_expr->get_output_port_descriptor(0);
    const auto& new_desc_1 = new_expr->get_output_port_descriptor(1);

    EXPECT_EQ(new_desc_0->get_subtensor(), expected_desc_0->get_subtensor());
    EXPECT_EQ(new_desc_1->get_subtensor(), expected_desc_1->get_subtensor());
    EXPECT_EQ(new_desc_0->get_layout(), expected_desc_0->get_layout());
    EXPECT_EQ(new_desc_1->get_layout(), expected_desc_1->get_layout());
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
