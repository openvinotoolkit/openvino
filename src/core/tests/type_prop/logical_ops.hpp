// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

template <typename T, ngraph::element::Type_t ELEMENT_TYPE>
class LogicalOperatorType
{
public:
    using op_type = T;
    static constexpr ngraph::element::Type_t element_type = ELEMENT_TYPE;
};

template <typename T>
class LogicalOperatorTypeProp : public testing::Test
{
};

class LogicalOperatorTypeName
{
public:
    template <typename T>
    static std::string GetName(int)
    {
        using OP_Type = typename T::op_type;
        const ngraph::Node::type_info_t typeinfo = OP_Type::get_type_info_static();
        return typeinfo.name;
    }
};

TYPED_TEST_SUITE_P(LogicalOperatorTypeProp);

namespace
{
    template <typename T>
    void incorrect_init(const ngraph::element::Type& type,
                        const std::string& err,
                        const ngraph::Shape& shape1 = {1, 3, 6},
                        const ngraph::Shape& shape2 = {1, 3, 6})
    {
        auto input1 = std::make_shared<ngraph::op::Parameter>(type, shape1);
        auto input2 = std::make_shared<ngraph::op::Parameter>(type, shape2);
        try
        {
            auto op = std::make_shared<T>(input1, input2);
        }
        catch (const ngraph::NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), err);
        }
    }
} // namespace

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_f32)
{
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(
        ngraph::element::f32,
        "Operands for logical operators must have boolean element type but have element type f32");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_f64)
{
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(
        ngraph::element::f64,
        "Operands for logical operators must have boolean element type but have element type f64");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_i32)
{
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(
        ngraph::element::i32,
        "Operands for logical operators must have boolean element type but have element type i32");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_i64)
{
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(
        ngraph::element::i64,
        "Operands for logical operators must have boolean element type but have element type i64");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_u32)
{
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(
        ngraph::element::u32,
        "Operands for logical operators must have boolean element type but have element type u32");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_u64)
{
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(
        ngraph::element::u64,
        "Operands for logical operators must have boolean element type but have element type u64");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_shape)
{
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(ngraph::element::boolean,
                            "Argument shapes are inconsistent",
                            ngraph::Shape{1, 3, 6},
                            ngraph::Shape{1, 2, 3});
}

TYPED_TEST_P(LogicalOperatorTypeProp, broadcast)
{
    using OP_Type = typename TypeParam::op_type;

    auto input1 =
        std::make_shared<ngraph::op::Parameter>(ngraph::element::boolean, ngraph::Shape{1, 1, 6});
    auto input2 =
        std::make_shared<ngraph::op::Parameter>(ngraph::element::boolean, ngraph::Shape{1, 3, 1});

    auto logical_and = std::make_shared<OP_Type>(input1, input2);

    ASSERT_EQ(logical_and->get_element_type(), ngraph::element::boolean);
    ASSERT_EQ(logical_and->get_shape(), (ngraph::Shape{1, 3, 6}));
}

REGISTER_TYPED_TEST_SUITE_P(LogicalOperatorTypeProp,
                            broadcast,
                            incorrect_type_f32,
                            incorrect_type_f64,
                            incorrect_type_i32,
                            incorrect_type_i64,
                            incorrect_type_u32,
                            incorrect_type_u64,
                            incorrect_shape);
