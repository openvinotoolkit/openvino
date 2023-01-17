// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "ngraph_functions/builders.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, prior_box_op) {
    NodeBuilder::get_ops().register_factory<opset1::PriorBox>();
    const auto layer_shape = make_shared<op::Parameter>(element::i64, Shape{128, 128});
    const auto image_shape = make_shared<op::Parameter>(element::i64, Shape{32, 32});

    op::v0::PriorBox::Attributes attrs;
    attrs.min_size = vector<float>{16.f, 32.f};
    attrs.max_size = vector<float>{256.f, 512.f};
    attrs.aspect_ratio = vector<float>{0.66f, 1.56f};
    attrs.density = vector<float>{0.55f};
    attrs.fixed_ratio = vector<float>{0.88f};
    attrs.fixed_size = vector<float>{1.25f};
    attrs.clip = true;
    attrs.flip = false;
    attrs.step = 1.0f;
    attrs.offset = 0.0f;
    attrs.variance = vector<float>{2.22f, 3.14f};
    attrs.scale_all_sizes = true;

    auto prior_box = make_shared<opset1::PriorBox>(layer_shape, image_shape, attrs);
    NodeBuilder builder(prior_box, {layer_shape, image_shape});
    auto g_prior_box = ov::as_type_ptr<opset1::PriorBox>(builder.create());

    const auto prior_box_attrs = prior_box->get_attrs();
    const auto g_prior_box_attrs = g_prior_box->get_attrs();

    const auto expected_attr_count = 12;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_prior_box_attrs.min_size, prior_box_attrs.min_size);
    EXPECT_EQ(g_prior_box_attrs.max_size, prior_box_attrs.max_size);
    EXPECT_EQ(g_prior_box_attrs.aspect_ratio, prior_box_attrs.aspect_ratio);
    EXPECT_EQ(g_prior_box_attrs.density, prior_box_attrs.density);
    EXPECT_EQ(g_prior_box_attrs.fixed_ratio, prior_box_attrs.fixed_ratio);
    EXPECT_EQ(g_prior_box_attrs.fixed_size, prior_box_attrs.fixed_size);
    EXPECT_EQ(g_prior_box_attrs.clip, prior_box_attrs.clip);
    EXPECT_EQ(g_prior_box_attrs.flip, prior_box_attrs.flip);
    EXPECT_EQ(g_prior_box_attrs.step, prior_box_attrs.step);
    EXPECT_EQ(g_prior_box_attrs.offset, prior_box_attrs.offset);
    EXPECT_EQ(g_prior_box_attrs.variance, prior_box_attrs.variance);
    EXPECT_EQ(g_prior_box_attrs.scale_all_sizes, prior_box_attrs.scale_all_sizes);
    EXPECT_EQ(g_prior_box->has_evaluate(), prior_box->has_evaluate());
}

TEST(attributes, prior_box_op2) {
    NodeBuilder::get_ops().register_factory<opset1::PriorBox>();
    const auto layer_shape = make_shared<op::Parameter>(element::i64, Shape{2});
    const auto image_shape = make_shared<op::Parameter>(element::i64, Shape{2});

    op::v0::PriorBox::Attributes attrs;
    attrs.min_size = vector<float>{0.1f, 0.141421f};
    attrs.max_size = vector<float>{};
    attrs.aspect_ratio = vector<float>{2.0f, 0.5f};
    attrs.density = vector<float>{};
    attrs.fixed_ratio = vector<float>{};
    attrs.fixed_size = vector<float>{};
    attrs.clip = false;
    attrs.flip = false;
    attrs.step = 0.03333333f;
    attrs.offset = 0.5f;
    attrs.variance = vector<float>{0.1f, 0.1f, 0.2f, 0.2f};
    attrs.scale_all_sizes = false;

    auto prior_box = make_shared<opset1::PriorBox>(layer_shape, image_shape, attrs);
    NodeBuilder builder(prior_box, {layer_shape, image_shape});
    auto g_prior_box = ov::as_type_ptr<opset1::PriorBox>(builder.create());

    const auto prior_box_attrs = prior_box->get_attrs();
    const auto g_prior_box_attrs = g_prior_box->get_attrs();

    const auto expected_attr_count = 12;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_prior_box_attrs.min_size, prior_box_attrs.min_size);
    EXPECT_EQ(g_prior_box_attrs.max_size, prior_box_attrs.max_size);
    EXPECT_EQ(g_prior_box_attrs.aspect_ratio, prior_box_attrs.aspect_ratio);
    EXPECT_EQ(g_prior_box_attrs.density, prior_box_attrs.density);
    EXPECT_EQ(g_prior_box_attrs.fixed_ratio, prior_box_attrs.fixed_ratio);
    EXPECT_EQ(g_prior_box_attrs.fixed_size, prior_box_attrs.fixed_size);
    EXPECT_EQ(g_prior_box_attrs.clip, prior_box_attrs.clip);
    EXPECT_EQ(g_prior_box_attrs.flip, prior_box_attrs.flip);
    EXPECT_EQ(g_prior_box_attrs.step, prior_box_attrs.step);
    EXPECT_EQ(g_prior_box_attrs.offset, prior_box_attrs.offset);
    EXPECT_EQ(g_prior_box_attrs.variance, prior_box_attrs.variance);
    EXPECT_EQ(g_prior_box_attrs.scale_all_sizes, prior_box_attrs.scale_all_sizes);
    EXPECT_EQ(g_prior_box->has_evaluate(), prior_box->has_evaluate());
}

TEST(attributes, prior_box_v8_op) {
    NodeBuilder::get_ops().register_factory<opset8::PriorBox>();
    const auto layer_shape = make_shared<op::Parameter>(element::i64, Shape{128, 128});
    const auto image_shape = make_shared<op::Parameter>(element::i64, Shape{32, 32});

    op::v8::PriorBox::Attributes attrs;
    attrs.min_size = vector<float>{16.f, 32.f};
    attrs.max_size = vector<float>{256.f, 512.f};
    attrs.aspect_ratio = vector<float>{0.66f, 1.56f};
    attrs.density = vector<float>{0.55f};
    attrs.fixed_ratio = vector<float>{0.88f};
    attrs.fixed_size = vector<float>{1.25f};
    attrs.clip = true;
    attrs.flip = false;
    attrs.step = 1.0f;
    attrs.offset = 0.0f;
    attrs.variance = vector<float>{2.22f, 3.14f};
    attrs.scale_all_sizes = true;
    attrs.min_max_aspect_ratios_order = false;

    auto prior_box = make_shared<opset8::PriorBox>(layer_shape, image_shape, attrs);
    NodeBuilder builder(prior_box, {layer_shape, image_shape});
    auto g_prior_box = ov::as_type_ptr<opset8::PriorBox>(builder.create());

    const auto prior_box_attrs = prior_box->get_attrs();
    const auto g_prior_box_attrs = g_prior_box->get_attrs();

    const auto expected_attr_count = 13;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_prior_box_attrs.min_size, prior_box_attrs.min_size);
    EXPECT_EQ(g_prior_box_attrs.max_size, prior_box_attrs.max_size);
    EXPECT_EQ(g_prior_box_attrs.aspect_ratio, prior_box_attrs.aspect_ratio);
    EXPECT_EQ(g_prior_box_attrs.density, prior_box_attrs.density);
    EXPECT_EQ(g_prior_box_attrs.fixed_ratio, prior_box_attrs.fixed_ratio);
    EXPECT_EQ(g_prior_box_attrs.fixed_size, prior_box_attrs.fixed_size);
    EXPECT_EQ(g_prior_box_attrs.clip, prior_box_attrs.clip);
    EXPECT_EQ(g_prior_box_attrs.flip, prior_box_attrs.flip);
    EXPECT_EQ(g_prior_box_attrs.step, prior_box_attrs.step);
    EXPECT_EQ(g_prior_box_attrs.offset, prior_box_attrs.offset);
    EXPECT_EQ(g_prior_box_attrs.variance, prior_box_attrs.variance);
    EXPECT_EQ(g_prior_box_attrs.scale_all_sizes, prior_box_attrs.scale_all_sizes);
    EXPECT_EQ(g_prior_box_attrs.min_max_aspect_ratios_order, prior_box_attrs.min_max_aspect_ratios_order);
    EXPECT_EQ(g_prior_box->has_evaluate(), prior_box->has_evaluate());
}

TEST(attributes, prior_box_v8_op2) {
    NodeBuilder::get_ops().register_factory<opset8::PriorBox>();

    auto params = ngraph::builder::makeParams(ov::element::Type_t::i32, {{128, 128}, {32, 32}});
    auto shape_of_1 = std::make_shared<ngraph::opset3::ShapeOf>(params[0]);
    auto shape_of_2 = std::make_shared<ngraph::opset3::ShapeOf>(params[1]);

    op::v8::PriorBox::Attributes attrs;
    attrs.min_size = vector<float>{16.f, 32.f};
    attrs.max_size = vector<float>{256.f, 512.f};
    attrs.aspect_ratio = vector<float>{0.66f, 1.56f};
    attrs.density = vector<float>{0.55f};
    attrs.fixed_ratio = vector<float>{0.88f};
    attrs.fixed_size = vector<float>{1.25f};
    attrs.clip = true;
    attrs.flip = false;
    attrs.step = 1.0f;
    attrs.offset = 0.0f;
    attrs.variance = vector<float>{2.22f, 3.14f};
    attrs.scale_all_sizes = true;
    attrs.min_max_aspect_ratios_order = false;

    auto prior_box = make_shared<opset8::PriorBox>(shape_of_1, shape_of_2, attrs);
    NodeBuilder builder(prior_box, {shape_of_1, shape_of_2});
    auto g_prior_box = ov::as_type_ptr<opset8::PriorBox>(builder.create());

    const auto prior_box_attrs = prior_box->get_attrs();
    const auto g_prior_box_attrs = g_prior_box->get_attrs();

    const auto expected_attr_count = 13;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_prior_box_attrs.min_size, prior_box_attrs.min_size);
    EXPECT_EQ(g_prior_box_attrs.max_size, prior_box_attrs.max_size);
    EXPECT_EQ(g_prior_box_attrs.aspect_ratio, prior_box_attrs.aspect_ratio);
    EXPECT_EQ(g_prior_box_attrs.density, prior_box_attrs.density);
    EXPECT_EQ(g_prior_box_attrs.fixed_ratio, prior_box_attrs.fixed_ratio);
    EXPECT_EQ(g_prior_box_attrs.fixed_size, prior_box_attrs.fixed_size);
    EXPECT_EQ(g_prior_box_attrs.clip, prior_box_attrs.clip);
    EXPECT_EQ(g_prior_box_attrs.flip, prior_box_attrs.flip);
    EXPECT_EQ(g_prior_box_attrs.step, prior_box_attrs.step);
    EXPECT_EQ(g_prior_box_attrs.offset, prior_box_attrs.offset);
    EXPECT_EQ(g_prior_box_attrs.variance, prior_box_attrs.variance);
    EXPECT_EQ(g_prior_box_attrs.scale_all_sizes, prior_box_attrs.scale_all_sizes);
    EXPECT_EQ(g_prior_box_attrs.min_max_aspect_ratios_order, prior_box_attrs.min_max_aspect_ratios_order);
    EXPECT_EQ(g_prior_box->has_evaluate(), prior_box->has_evaluate());
}
