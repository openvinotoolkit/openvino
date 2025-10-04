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

namespace {
ov::snippets::VectorDims make_dims(std::initializer_list<size_t> values) {
    return ov::snippets::VectorDims(values);
}
}  // namespace

TEST(LinearIRReplaceWithNode, PreservesPerOutputDescriptors) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 6});
    const auto op = std::make_shared<ov::snippets::op::OnlineSoftmax>(param);
    const auto result_0 = std::make_shared<ov::op::v0::Result>(op->output(0));
    const auto result_1 = std::make_shared<ov::op::v0::Result>(op->output(1));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{result_0, result_1}, ov::ParameterVector{param});

    auto factory = std::make_shared<ov::snippets::IShapeInferSnippetsFactory>();
    ov::snippets::lowered::LinearIR linear_ir(model, factory);

    const auto& split_expr = linear_ir.get_expr_by_node(op);
    ASSERT_NE(nullptr, split_expr);

    init_expr_descriptors(split_expr);

    split_expr->get_output_port_descriptor(0)->set_subtensor(make_dims({1, 3}));
    split_expr->get_output_port_descriptor(0)->set_layout(make_dims({0, 1}));
    split_expr->get_output_port_descriptor(1)->set_subtensor(make_dims({1, 6}));
    split_expr->get_output_port_descriptor(1)->set_layout(make_dims({1, 0}));

    const auto expected_desc_0 = split_expr->get_output_port_descriptor(0)->clone();
    const auto expected_desc_1 = split_expr->get_output_port_descriptor(1)->clone();

    ASSERT_NE(expected_desc_0->get_subtensor(), expected_desc_1->get_subtensor());
    ASSERT_NE(expected_desc_0->get_layout(), expected_desc_1->get_layout());

    const auto new_node = std::make_shared<ov::snippets::op::OnlineSoftmax>(param);
    linear_ir.replace_with_node({split_expr}, new_node);

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
