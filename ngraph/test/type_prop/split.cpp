//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(type_prop, split)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});

    try
    {
        const auto axis = op::Constant::create(element::i64, Shape{}, {1});
        const auto split = make_shared<op::v1::Split>(data, axis, 7);
        FAIL() << "Split node was created with incorrect data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("The input tensor's dimension pointed by the 'axis' parameter: 6 has to be "
                        "a multiple of the 'num_splits' attribute value: 7"));
    }

    try
    {
        const auto axis = op::Constant::create(element::i64, Shape{}, {-5});
        const auto split = make_shared<op::v1::Split>(data, axis, 4); // invalid axis
        FAIL() << "Split node was created with incorrect data.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis -5 out of the tensor rank"));
    }

    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    const auto split = make_shared<op::v1::Split>(data, axis, 2);
    EXPECT_EQ(split->outputs().size(), 2);
    EXPECT_EQ(split->get_output_shape(0), (Shape{2, 3}));
    EXPECT_EQ(split->get_output_shape(1), (Shape{2, 3}));
    EXPECT_EQ(split->get_output_element_type(0), element::i32);
    EXPECT_EQ(split->get_output_element_type(1), element::i32);
}

TEST(type_prop, split_axis_must_be_scalar)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const auto axis = op::Constant::create(element::i64, Shape{2}, {0, 1});

    try
    {
        const auto split = make_shared<op::v1::Split>(data, axis, 1);
        FAIL() << "Incorrect axis of Split not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The 'axis' input is expected to be a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason.";
    }
}

TEST(type_prop, split_v1)
{
    const auto data = make_shared<op::Parameter>(element::f16, Shape{2, 3, 4});
    const auto axis = op::Constant::create(element::i64, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (int i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_element_type(i), element::f16);
        EXPECT_EQ(split->get_output_shape(i), (Shape{2, 1, 4}));
    }
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_known)
{
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    const auto axis = op::Constant::create(element::i32, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (int i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), (PartialShape{2, 1, Dimension::dynamic()}));
    }
}

TEST(type_prop, split_v1_axis_const_only_data_axis_dim_known)
{
    const auto data = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    const auto axis = op::Constant::create(element::i16, {}, {0});
    const size_t num_splits = 2;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (int i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i),
                  (PartialShape{1, Dimension::dynamic(), Dimension::dynamic()}));
    }
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_unknown)
{
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{4, Dimension::dynamic(), 3, 5});
    const auto axis = op::Constant::create(element::i8, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (int i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i),
                  (PartialShape{4, Dimension::dynamic(), 3, 5}));
    }
}

TEST(type_prop, split_v1_axis_const_only_data_rank_known)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto axis = op::Constant::create(element::u64, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (int i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic(4));
    }
}

TEST(type_prop, split_v1_axis_not_const_only_data_rank_known)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto axis = make_shared<op::Parameter>(element::u32, PartialShape{});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (int i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic(4));
    }
}

TEST(type_prop, split_v1_axis_const_data_rank_unknown)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = op::Constant::create(element::u16, {}, {2});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (int i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic());
    }
}

TEST(type_prop, split_v1_axis_not_const_data_rank_unknown)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = make_shared<op::Parameter>(element::u8, PartialShape{});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (int i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic());
    }
}

TEST(type_prop, split_v1_axis_dynamic_rank)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = make_shared<op::Parameter>(element::u8, PartialShape::dynamic());
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (int i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic());
    }
}
