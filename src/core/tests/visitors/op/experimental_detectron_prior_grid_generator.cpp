// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_prior_grid_generator.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

using ExperimentalGenerator = ov::op::v6::ExperimentalDetectronPriorGridGenerator;
using Attrs = ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes;

TEST(attributes, detectron_prior_grid_generator) {
    NodeBuilder::opset().insert<ExperimentalGenerator>();

    Attrs attrs;
    attrs.flatten = true;
    attrs.h = 3;
    attrs.w = 6;
    attrs.stride_x = 64;
    attrs.stride_y = 64;

    auto priors = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});
    auto feature_map = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 100, 100});
    auto im_data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 100, 200});

    auto proposals = std::make_shared<ExperimentalGenerator>(priors, feature_map, im_data, attrs);

    NodeBuilder builder(proposals, {priors, feature_map, im_data});

    auto g_proposals = ov::as_type_ptr<ExperimentalGenerator>(builder.create());

    const auto expected_attr_count = 5;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_proposals->get_attrs().flatten, proposals->get_attrs().flatten);
    EXPECT_EQ(g_proposals->get_attrs().h, proposals->get_attrs().h);
    EXPECT_EQ(g_proposals->get_attrs().w, proposals->get_attrs().w);
    EXPECT_EQ(g_proposals->get_attrs().stride_x, proposals->get_attrs().stride_x);
    EXPECT_EQ(g_proposals->get_attrs().stride_y, proposals->get_attrs().stride_y);
}
