// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, logical_and_incorrect_type_f32)
{
    auto input1 = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto input2 = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    try
    {
        auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Operands for logical operators must have boolean element type but have element type f32"));
    }

}

TEST(type_prop, logical_and_incorrect_type_f64)
{
    auto input1 = make_shared<op::Parameter>(element::f64, Shape{1, 3, 6});
    auto input2 = make_shared<op::Parameter>(element::f64, Shape{1, 3, 6});
    try
    {
        auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Operands for logical operators must have boolean element type but have element type f64"));
    }
}

TEST(type_prop, logical_and_incorrect_type_i32)
{
    auto input1 = make_shared<op::Parameter>(element::i32, Shape{1, 3, 6});
    auto input2 = make_shared<op::Parameter>(element::i32, Shape{1, 3, 6});
    try
    {
        auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Operands for logical operators must have boolean element type but have element type i32"));
    }
}

TEST(type_prop, logical_and_incorrect_type_i64)
{
    auto input1 = make_shared<op::Parameter>(element::i64, Shape{1, 3, 6});
    auto input2 = make_shared<op::Parameter>(element::i64, Shape{1, 3, 6});
    try
    {
        auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Operands for logical operators must have boolean element type but have element type i64"));
    }
}

TEST(type_prop, logical_and_incorrect_type_u32)
{
    auto input1 = make_shared<op::Parameter>(element::u32, Shape{1, 3, 6});
    auto input2 = make_shared<op::Parameter>(element::u32, Shape{1, 3, 6});
    try
    {
        auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Operands for logical operators must have boolean element type but have element type u32"));
    }
}

TEST(type_prop, logical_and_incorrect_type_u64)
{
    auto input1 = make_shared<op::Parameter>(element::u64, Shape{1, 3, 6});
    auto input2 = make_shared<op::Parameter>(element::u64, Shape{1, 3, 6});
    try
    {
        auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Operands for logical operators must have boolean element type but have element type u64"));
    }
}

TEST(type_prop, logical_and_incorrect_shape)
{
    auto input1 = make_shared<op::Parameter>(element::boolean, Shape{1, 3, 6});
    auto input2 = make_shared<op::Parameter>(element::boolean, Shape{1, 2, 6});
    try
    {
        auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent"));
    }
}

TEST(type_prop, logical_and_broadcast)
{
    auto input1 = make_shared<op::Parameter>(element::boolean, Shape{1, 1, 6});
    auto input2 = make_shared<op::Parameter>(element::boolean, Shape{1, 3, 1});

    auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);

    ASSERT_EQ(logical_and->get_element_type(), element::boolean);
    ASSERT_EQ(logical_and->get_shape(), (Shape{1, 3, 6}));
}
