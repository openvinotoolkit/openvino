// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, squeeze)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 4, 1, 8});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{0, 2});
    auto squeeze = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);
    ASSERT_EQ(squeeze->get_shape(), (Shape{4, 4, 1, 8}));

    axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{0}, vector<int64_t>{});
    auto squeeze_default_axes = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze_default_axes->get_element_type(), element::f32);
    ASSERT_EQ(squeeze_default_axes->get_shape(), (Shape{4, 4, 8}));
}

TEST(type_prop, squeeze_unsqueezable_no_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(2, 5), Dimension(3, 4), 6});
    auto squeeze = make_shared<op::Squeeze>(param);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_TRUE(squeeze->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension(2, 5), Dimension(3, 4), 6}));
}

TEST(type_prop, squeeze_no_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 4, 1, 8});
    auto squeeze = make_shared<op::Squeeze>(param);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);
    ASSERT_EQ(squeeze->get_shape(), (Shape{4, 4, 8}));

    auto axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{0}, vector<int64_t>{});
    auto squeeze_default_axes = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze_default_axes->get_element_type(), element::f32);
    ASSERT_EQ(squeeze_default_axes->get_shape(), (Shape{4, 4, 8}));
}

TEST(type_prop, squeeze_dynamic_static_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(6));
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{0, 2});
    auto squeeze = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);

    EXPECT_TRUE(squeeze->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));

    axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{0}, vector<int64_t>{});
    auto squeeze_default_axes = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze_default_axes->get_element_type(), element::f32);
    EXPECT_TRUE(
        squeeze_default_axes->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, squeeze_dynamic_dynamic_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{0, 2});
    auto squeeze = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);

    EXPECT_TRUE(squeeze->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));

    axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{0}, vector<int64_t>{});
    auto squeeze_default_axes = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze_default_axes->get_element_type(), element::f32);
    EXPECT_TRUE(
        squeeze_default_axes->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, squeeze_axes_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 4, 1, 8});
    auto axes_node =
        make_shared<ngraph::op::Parameter>(element::u64, PartialShape::dynamic());
    auto squeeze = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);
    ASSERT_TRUE(squeeze->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, squeeze_axes_invalid_value)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{0, 2});

    try
    {
        auto squeeze = make_shared<op::Squeeze>(param, axes_node);
        FAIL() << "Squeeze axis invalid value not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "provided axis value is invalid. Only axes of size 1 may be removed.");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, squeeze_axes_invalid_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::i32, Shape{2, 1}, vector<int32_t>{0, 2});

    try
    {
        auto squeeze = make_shared<op::Squeeze>(param, axes_node);
        FAIL() << "Squeeze axis invalid rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Second input (axes) should not be of rank higher than 1.");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, squeeze_negative_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 4, 1, 8});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::i64, Shape{2}, vector<int64_t>{-6, -4});
    auto squeeze = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);
    ASSERT_EQ(squeeze->get_shape(), (Shape{4, 4, 1, 8}));

    axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{0}, vector<int64_t>{});
    auto squeeze_default_axes = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze_default_axes->get_element_type(), element::f32);
    ASSERT_EQ(squeeze_default_axes->get_shape(), (Shape{4, 4, 8}));
}


TEST(type_prop, squeeze_incorrect_negative_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 4, 1, 8});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::i64, Shape{2}, vector<int64_t>{-6, -10});

    try     
    {
        auto squeeze = make_shared<op::Squeeze>(param, axes_node);
        FAIL() << "Squeeze axis invalid value not detected";
    }
    catch (ngraph_error &error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Parameter axis -10 out of the tensor rank range");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, squeeze_scalar_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 4, 1, 8});
    auto axes_node =
            make_shared<ngraph::op::Constant>(element::i64, Shape{}, vector<int64_t>{2});
    auto squeeze = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);
    ASSERT_EQ(squeeze->get_shape(), (Shape{1, 4, 4, 1, 8}));

    int squeeze_index = 0;
    axes_node = make_shared<ngraph::op::Constant>(element::i64, Shape{}, squeeze_index);
    squeeze = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);
    ASSERT_EQ(squeeze->get_shape(), (Shape{4, 1, 4, 1, 8}));
}
