// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, shape_of_v0)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_et_dynamic_v0)
{
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_static_dynamic_v0)
{
    auto a = make_shared<op::Parameter>(
        element::f32, PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 4});
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_dynamic_v0)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_TRUE(so->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, shape_of_v3)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_et_dynamic_v3)
{
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_static_dynamic_v3)
{
    auto a = make_shared<op::Parameter>(
        element::f32, PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 4});
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_dynamic_v3)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_TRUE(so->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, shape_of_output_type_v3)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v3::ShapeOf>(a, element::i32);
    try
    {
        auto sx = make_shared<op::v3::ShapeOf>(a, element::i8);
        FAIL() << "Invalid output_type not detected";
    }
    catch (const NodeValidationFailure&)
    {
    }
    catch (...)
    {
        FAIL() << "Node validation error not thrown";
    }
    try
    {
        auto sx = make_shared<op::v3::ShapeOf>(a, element::i16);
        FAIL() << "Invalid output_type not detected";
    }
    catch (const NodeValidationFailure&)
    {
    }
    catch (...)
    {
        FAIL() << "Node validation error not thrown";
    }
    try
    {
        auto sx = make_shared<op::v3::ShapeOf>(a, element::f32);
        FAIL() << "Invalid output_type not detected";
    }
    catch (const NodeValidationFailure&)
    {
    }
    catch (...)
    {
        FAIL() << "Node validation error not thrown";
    }

    ASSERT_EQ(so->get_output_element_type(0), element::i32);
    ASSERT_EQ(so->get_shape(), Shape{4});
}
