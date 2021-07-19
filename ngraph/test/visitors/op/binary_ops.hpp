// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"
#include "util/visitor.hpp"

template <typename T, ngraph::element::Type_t ELEMENT_TYPE>
class BinaryOperatorType
{
public:
    using op_type = T;
    static constexpr ngraph::element::Type_t element_type = ELEMENT_TYPE;
};

template <typename T>
class BinaryOperatorVisitor : public testing::Test
{
};

TYPED_TEST_SUITE_P(BinaryOperatorVisitor);

TYPED_TEST_P(BinaryOperatorVisitor, No_Attribute_4D)
{
    using OP_Type = typename TypeParam::op_type;
    const ngraph::element::Type_t element_type = TypeParam::element_type;

    ngraph::test::NodeBuilder::get_ops().register_factory<OP_Type>();
    const auto A =
        std::make_shared<ngraph::op::Parameter>(element_type, ngraph::PartialShape{1, 2, 3});
    const auto B =
        std::make_shared<ngraph::op::Parameter>(element_type, ngraph::PartialShape{3, 2, 1});

    auto auto_broadcast = ngraph::op::AutoBroadcastType::NUMPY;

    const auto op_func = std::make_shared<OP_Type>(A, B, auto_broadcast);
    ngraph::test::NodeBuilder builder(op_func);
    const auto g_op_func = ngraph::as_type_ptr<OP_Type>(builder.create());

    EXPECT_EQ(op_func->get_autob(), g_op_func->get_autob());
}

REGISTER_TYPED_TEST_SUITE_P(BinaryOperatorVisitor, No_Attribute_4D);
