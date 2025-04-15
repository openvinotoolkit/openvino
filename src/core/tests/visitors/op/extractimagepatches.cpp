// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/extractimagepatches.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, extractimagepatches_op) {
    NodeBuilder::opset().insert<ov::op::v3::ExtractImagePatches>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{64, 3, 10, 10});

    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = ov::op::PadType::VALID;

    auto extractimagepatches =
        make_shared<ov::op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);
    NodeBuilder builder(extractimagepatches, {data});
    auto g_extractimagepatches = ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(builder.create());

    const auto expected_attr_count = 4;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_extractimagepatches->get_sizes(), sizes);
    EXPECT_EQ(g_extractimagepatches->get_strides(), strides);
    EXPECT_EQ(g_extractimagepatches->get_rates(), rates);
    EXPECT_EQ(g_extractimagepatches->get_auto_pad(), padtype_padding);
}
