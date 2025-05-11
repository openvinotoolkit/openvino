// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

template <typename T, ov::element::Type_t ELEMENT_TYPE>
class BinaryOperatorType {
public:
    using op_type = T;
    static constexpr ov::element::Type_t element_type = ELEMENT_TYPE;
};

template <typename T>
class BinaryOperatorVisitor : public testing::Test {};

class BinaryOperatorTypeName {
public:
    template <typename T>
    static std::string GetName(int) {
        using OP_Type = typename T::op_type;
        constexpr ov::element::Type precision(T::element_type);
        const ov::Node::type_info_t typeinfo = OP_Type::get_type_info_static();
        return std::string{typeinfo.name} + "_" + precision.get_type_name();
    }
};

TYPED_TEST_SUITE_P(BinaryOperatorVisitor);

TYPED_TEST_P(BinaryOperatorVisitor, Auto_Broadcast) {
    using OP_Type = typename TypeParam::op_type;
    const ov::element::Type_t element_type = TypeParam::element_type;

    ov::test::NodeBuilder::opset().insert<OP_Type>();
    const auto A = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1, 2, 3});
    const auto B = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{3, 2, 1});

    auto auto_broadcast = ov::op::AutoBroadcastType::NUMPY;

    const auto op_func = std::make_shared<OP_Type>(A, B, auto_broadcast);
    ov::test::NodeBuilder builder(op_func, {A, B});
    const auto g_op_func = ov::as_type_ptr<OP_Type>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(op_func->get_autob(), g_op_func->get_autob());
}

TYPED_TEST_P(BinaryOperatorVisitor, No_Broadcast) {
    using OP_Type = typename TypeParam::op_type;
    const ov::element::Type_t element_type = TypeParam::element_type;

    ov::test::NodeBuilder::opset().insert<OP_Type>();
    const auto A = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1, 2, 3});
    const auto B = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1, 2, 3});

    const auto op_func = std::make_shared<OP_Type>(A, B);
    ov::test::NodeBuilder builder(op_func, {A, B});
    const auto g_op_func = ov::as_type_ptr<OP_Type>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(op_func->get_autob(), g_op_func->get_autob());
}

REGISTER_TYPED_TEST_SUITE_P(BinaryOperatorVisitor, Auto_Broadcast, No_Broadcast);
