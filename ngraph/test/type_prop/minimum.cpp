// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, minimum_2D_same)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, minimum_4D_same)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_shape(), (Shape{2, 2, 3, 3}));
}

TEST(type_prop, minimum_default_autobroadcast)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_shape(), (Shape{2, 2}));
    ASSERT_EQ(minimum->get_autob(), op::AutoBroadcastType::NUMPY);
}

TEST(type_prop, minimum_no_autobroadcast)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    auto minimum = make_shared<op::v1::Minimum>(A, B, op::AutoBroadcastSpec::NONE);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_shape(), (Shape{2, 2}));
    ASSERT_EQ(minimum->get_autob(), op::AutoBroadcastType::NONE);
}

TEST(type_prop, minimum_4D_x_scalar_numpy_broadcast)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_shape(), (Shape{2, 3, 4, 5}));
}

TEST(type_prop, minimum_4D_x_1D_numpy_broadcast)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto B = make_shared<op::Parameter>(element::f32, Shape{5});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_shape(), (Shape{2, 3, 4, 5}));
}

TEST(type_prop, minimum_2D_x_4D_numpy_broadcast)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_shape(), (Shape{2, 3, 4, 5}));
}

TEST(type_prop, minimum_3D_x_4D_numpy_broadcast)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 4, 5});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 3, 1, 1});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_shape(), (Shape{2, 3, 4, 5}));
}

TEST(type_prop, minimum_4D_x_3D_numpy_broadcast)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{8, 1, 6, 1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{7, 1, 5});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_shape(), (Shape{8, 7, 6, 5}));
    ASSERT_EQ(minimum->get_autob(), op::AutoBroadcastType::NUMPY);
}

TEST(type_prop, minimum_incompatible_element_types)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = make_shared<op::Parameter>(element::i32, Shape{2, 2, 3, 3});

    try
    {
        auto minimum = make_shared<op::v1::Minimum>(A, B);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible element types not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Minimum element type validation failed for unexpexted reason";
    }
}

TEST(type_prop, minimum_incompatible_boolean_type)
{
    auto A = make_shared<op::Parameter>(element::boolean, Shape{2, 2, 3, 3});
    auto B = make_shared<op::Parameter>(element::boolean, Shape{2, 2, 3, 3});

    try
    {
        auto minimum = make_shared<op::v1::Minimum>(A, B);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible boolean type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments cannot have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Minimum element type validation failed for unexpexted reason";
    }
}

TEST(type_prop, minimum_1D_x_1D_incompatible)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{4});

    try
    {
        auto minimum = make_shared<op::v1::Minimum>(A, B);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Minimum shape validation failed for unexpected reason";
    }
}

TEST(type_prop, minimum_3D_x_3D_incompatible)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 5, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{4, 10, 12});

    try
    {
        auto minimum = make_shared<op::v1::Minimum>(A, B);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Minimum shape validation failed for unexpected reason";
    }
}

TEST(type_prop, minimum_5D_x_5D_incompatible)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{389, 112, 12});
    auto B = make_shared<op::Parameter>(element::f32, Shape{389, 112, 19});

    try
    {
        auto minimum = make_shared<op::v1::Minimum>(A, B);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Minimum shape validation failed for unexpected reason";
    }
}

TEST(type_prop, minimum_3D_dynamic_shape)
{
    Dimension dynamic = Dimension::dynamic();
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{dynamic, dynamic, 6});
    auto B = make_shared<op::Parameter>(element::f32, PartialShape{dynamic, dynamic, 6});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_output_partial_shape(0), (PartialShape{dynamic, dynamic, 6}));
}

TEST(type_prop, minimum_5D_dynamic_shape)
{
    Dimension dynamic = Dimension::dynamic();
    auto A =
        make_shared<op::Parameter>(element::f32, PartialShape{dynamic, 4, dynamic, dynamic, 6});
    auto B =
        make_shared<op::Parameter>(element::f32, PartialShape{dynamic, 4, dynamic, dynamic, 6});

    auto minimum = make_shared<op::v1::Minimum>(A, B);

    ASSERT_EQ(minimum->get_element_type(), element::f32);
    ASSERT_EQ(minimum->get_output_partial_shape(0),
              (PartialShape{dynamic, 4, dynamic, dynamic, 6}));
}

TEST(type_prop, minimum_full_dynamic_shape)
{
    auto param = std::make_shared<op::Parameter>(element::f64, PartialShape::dynamic());
    const auto op = std::make_shared<op::v1::Minimum>(param, param);
    ASSERT_EQ(op->get_element_type(), element::f64);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
