// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, negative_incompatible_input_type_u16)
{
    auto param = make_shared<op::Parameter>(element::u16, Shape{2, 2, 2});

    try
    {
        auto top = make_shared<op::v0::Negative>(param);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input type has to be signed"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, negative_incompatible_input_type_u32)
{
    auto param = make_shared<op::Parameter>(element::u32, Shape{2, 2, 2});

    try
    {
        auto top = make_shared<op::v0::Negative>(param);

    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input type has to be signed"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, negative_incompatible_input_type_u64)
{
    auto param = make_shared<op::Parameter>(element::u64, Shape{2, 2, 2});

    try
    {
        auto top = make_shared<op::v0::Negative>(param);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input type has to be signed"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, negative_incompatible_input_type_boolean)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 2, 2});

    try
    {
        auto top = make_shared<op::v0::Negative>(param);

    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input type has to be signed"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, negative_inference_shape)
{
    auto param = std::make_shared<op::Parameter>(element::i32, Shape{21, 15, 2});
    auto op = std::make_shared<op::Negative>(param);
    ASSERT_EQ(op->get_shape(), (Shape{21, 15, 2}));
}

TEST(type_prop, negative_input_type_inference_i32)
{
    auto param = std::make_shared<op::Parameter>(element::i32, Shape{21, 15, 2});
    auto op = std::make_shared<op::Negative>(param);
    ASSERT_EQ(op->get_element_type(), element::i32);
}

TEST(type_prop, negative_input_type_inference_i64)
{
    auto param = std::make_shared<op::Parameter>(element::i64, Shape{21, 15, 2});
    auto op = std::make_shared<op::Negative>(param);
    ASSERT_EQ(op->get_element_type(), element::i64);
}

TEST(type_prop, negative_input_type_inference_f32)
{
    auto param = std::make_shared<op::Parameter>(element::f32, Shape{21, 15, 2});
    auto op = std::make_shared<op::Negative>(param);
    ASSERT_EQ(op->get_shape(), (Shape{21, 15, 2}));
}

TEST(type_prop, negative_input_type_inference_f64)
{
    auto param = std::make_shared<op::Parameter>(element::f32, Shape{21, 15, 2});
    auto op = std::make_shared<op::Negative>(param);
    ASSERT_EQ(op->get_shape(), (Shape{21, 15, 2}));
}

TEST(type_prop, dynamic_rank_input_shape_2D)
{
    const PartialShape param_shape{Dimension::dynamic(), 10};
    const auto param = std::make_shared<op::Parameter>(element::f32, param_shape);
    const auto op = std::make_shared<op::Negative>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(), 10}));
}

TEST(type_prop, dynamic_rank_input_shape_3D)
{
    const PartialShape param_shape{100, Dimension::dynamic(), 58};
    const auto param = std::make_shared<op::Parameter>(element::f32, param_shape);
    const auto op = std::make_shared<op::Negative>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{100, Dimension(), 58}));
}

TEST(type_prop, dynamic_rank_input_shape_full)
{
    const auto param = std::make_shared<op::Parameter>(element::f64, PartialShape::dynamic());
    const auto op = std::make_shared<op::Negative>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
