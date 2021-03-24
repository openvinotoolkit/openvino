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

TEST(type_prop, squeeze_dynamic)
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
