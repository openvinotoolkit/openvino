// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"
#include "util/visitor.hpp"

template <typename T, ngraph::element::Type_t ELEMENT_TYPE>
class UnaryOperatorType {
public:
    using op_type = T;
    static constexpr ngraph::element::Type_t element_type = ELEMENT_TYPE;
};
template <typename T>
class UnaryOperatorVisitor : public testing::Test {};

class UnaryOperatorTypeName {
public:
    template <typename T>
    static std::string GetName(int) {
        using OP_Type = typename T::op_type;
        constexpr ngraph::element::Type precision(T::element_type);
        const ngraph::Node::type_info_t typeinfo = OP_Type::get_type_info_static();
        return std::string{typeinfo.name} + "_" + precision.get_type_name();
    }
};

TYPED_TEST_SUITE_P(UnaryOperatorVisitor);

TYPED_TEST_P(UnaryOperatorVisitor, No_Attribute_4D) {
    using OP_Type = typename TypeParam::op_type;
    const ngraph::element::Type_t element_type = TypeParam::element_type;

    ngraph::test::NodeBuilder::get_ops().register_factory<OP_Type>();
    const auto A = std::make_shared<ngraph::op::Parameter>(element_type, ngraph::PartialShape{2, 2, 2, 2});

    const auto op_func = std::make_shared<OP_Type>(A);
    ngraph::test::NodeBuilder builder(op_func);
    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

REGISTER_TYPED_TEST_SUITE_P(UnaryOperatorVisitor, No_Attribute_4D);
