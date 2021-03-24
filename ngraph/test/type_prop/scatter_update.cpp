// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, scatter_update_v3_fail_indices_element_type)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::f16, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {1});
    try
    {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Indices element type must be of an integral number type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_update_v3_fail_updates_data_et_not_equal)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::u32, updates_shape);
    auto A = op::Constant::create(element::u32, Shape{1}, {1});
    try
    {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Element types for input data and updates do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_update_v3_fail_axis_element_type)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::i16, ref_shape);
    auto I = make_shared<op::Parameter>(element::u64, indices_shape);
    auto U = make_shared<op::Parameter>(element::i16, updates_shape);
    auto A = op::Constant::create(element::f32, Shape{1}, {1.5f});
    try
    {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Axis element type must be of an integral number type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_update_v3_fail_axis_shape)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::u8, ref_shape);
    auto I = make_shared<op::Parameter>(element::u16, indices_shape);
    auto U = make_shared<op::Parameter>(element::u8, updates_shape);
    auto A = op::Constant::create(element::u8, Shape{2}, {1, 5});
    try
    {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Axis input shape is required to be scalar or 1D tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_update_v3_fail_updates_rank)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 1, 4};
    auto R = make_shared<op::Parameter>(element::f64, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::f64, updates_shape);
    auto A = op::Constant::create(element::u8, Shape{}, {0});
    try
    {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Updates rank is expected to be indices rank + data rank - 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_axis)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::u64, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::u64, updates_shape);
    auto A = op::Constant::create(element::u16, Shape{}, {0});
    try
    {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Updates shape must have appropriate dimensions equal to indices and "
                        "data dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_indices)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 3, 1, 4};
    auto R = make_shared<op::Parameter>(element::u32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::u32, updates_shape);
    auto A = op::Constant::create(element::i32, Shape{}, {1});
    try
    {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Updates shape must have appropriate dimensions equal to indices and "
                        "data dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_data_before_axis)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{3, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::u16, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::u16, updates_shape);
    auto A = op::Constant::create(element::i8, Shape{}, {1});
    try
    {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Updates shape must have appropriate dimensions equal to indices and "
                        "data dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_data_after_axis)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 5};
    auto R = make_shared<op::Parameter>(element::i8, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::i8, updates_shape);
    auto A = op::Constant::create(element::i16, Shape{}, {1});
    try
    {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Updates shape must have appropriate dimensions equal to indices and "
                        "data dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_update_v3)
{
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::i8, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::i8, updates_shape);
    auto A = op::Constant::create(element::i16, Shape{}, {1});

    auto scatter_update = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
    EXPECT_EQ(scatter_update->get_output_element_type(0), element::i8);
    EXPECT_EQ(scatter_update->get_output_shape(0), ref_shape);
}

TEST(type_prop, scatter_update_v3_dynamic_data_shape)
{
    PartialShape ref_shape = PartialShape::dynamic();
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::i8, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::i8, updates_shape);
    auto A = op::Constant::create(element::i16, Shape{}, {1});

    auto scatter_update = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
    EXPECT_EQ(scatter_update->get_output_element_type(0), element::i8);
    EXPECT_TRUE(scatter_update->get_output_partial_shape(0).is_dynamic());
}
