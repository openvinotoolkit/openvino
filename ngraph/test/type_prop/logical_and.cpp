// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

namespace {
    void incorrect_init(const ngraph::element::Type& type, const std::string& err, const Shape& shape1 = {1, 3, 6}, const Shape& shape2 = {1, 3, 6}) {
        auto input1 = make_shared<op::Parameter>(type, shape1);
        auto input2 = make_shared<op::Parameter>(type, shape2);
        try
        {
            auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), err);
        }
    }
}

TEST(type_prop, logical_and_incorrect_type_f32)
{
    incorrect_init(element::f32, "Operands for logical operators must have boolean element type but have element type f32");
}

TEST(type_prop, logical_and_incorrect_type_f64)
{
    incorrect_init(element::f64, "Operands for logical operators must have boolean element type but have element type f64");
}

TEST(type_prop, logical_and_incorrect_type_i32)
{
    incorrect_init(element::i32, "Operands for logical operators must have boolean element type but have element type i32");
}

TEST(type_prop, logical_and_incorrect_type_i64)
{
    incorrect_init(element::i64, "Operands for logical operators must have boolean element type but have element type i64");
}

TEST(type_prop, logical_and_incorrect_type_u32)
{
    incorrect_init(element::u32, "Operands for logical operators must have boolean element type but have element type u32");
}

TEST(type_prop, logical_and_incorrect_type_u64)
{
    incorrect_init(element::u64, "Operands for logical operators must have boolean element type but have element type u64");

}

TEST(type_prop, logical_and_incorrect_shape)
{
    incorrect_init(element::boolean, "Argument shapes are inconsistent", Shape {1, 3, 6}, Shape {1, 2, 3});
}

TEST(type_prop, logical_and_broadcast)
{
    auto input1 = make_shared<op::Parameter>(element::boolean, Shape{1, 1, 6});
    auto input2 = make_shared<op::Parameter>(element::boolean, Shape{1, 3, 1});

    auto logical_and = make_shared<op::v1::LogicalAnd>(input1, input2);

    ASSERT_EQ(logical_and->get_element_type(), element::boolean);
    ASSERT_EQ(logical_and->get_shape(), (Shape{1, 3, 6}));
}
