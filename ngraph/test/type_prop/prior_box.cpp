// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/prior_box.hpp"

using namespace ngraph;

TEST(type_prop, prior_box1)
{
    op::PriorBoxAttrs attrs;
    attrs.min_size = {2.0f, 3.0f};
    attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
    attrs.scale_all_sizes = false;

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {32, 32});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
    auto pb = std::make_shared<op::PriorBox>(layer_shape, image_shape, attrs);
    ASSERT_EQ(pb->get_shape(), (Shape{2, 20480}));
}

TEST(type_prop, prior_box2)
{
    op::PriorBoxAttrs attrs;
    attrs.min_size = {2.0f, 3.0f};
    attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
    attrs.flip = true;
    attrs.scale_all_sizes = false;

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {32, 32});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
    auto pb = std::make_shared<op::PriorBox>(layer_shape, image_shape, attrs);
    ASSERT_EQ(pb->get_shape(), (Shape{2, 32768}));
}

TEST(type_prop, prior_box3)
{
    op::PriorBoxAttrs attrs;
    attrs.min_size = {256.0f};
    attrs.max_size = {315.0f};
    attrs.aspect_ratio = {2.0f};
    attrs.flip = true;

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 1});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
    auto pb = std::make_shared<op::PriorBox>(layer_shape, image_shape, attrs);
    ASSERT_EQ(pb->get_shape(), (Shape{2, 16}));
}