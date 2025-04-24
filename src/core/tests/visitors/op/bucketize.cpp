// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bucketize.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, bucketize_v3_op_default_attributes) {
    NodeBuilder::opset().insert<ov::op::v3::Bucketize>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    auto buckets = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});
    auto bucketize = make_shared<ov::op::v3::Bucketize>(data, buckets);
    NodeBuilder builder(bucketize, {data, buckets});

    auto g_bucketize = ov::as_type_ptr<ov::op::v3::Bucketize>(builder.create());

    EXPECT_EQ(g_bucketize->get_output_type(), bucketize->get_output_type());
    EXPECT_EQ(g_bucketize->get_with_right_bound(), bucketize->get_with_right_bound());
}

TEST(attributes, bucketize_v3_op_custom_attributes) {
    NodeBuilder::opset().insert<ov::op::v3::Bucketize>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    auto buckets = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});
    element::Type output_type = element::i32;
    bool with_right_bound = false;

    auto bucketize = make_shared<ov::op::v3::Bucketize>(data, buckets, output_type, with_right_bound);
    NodeBuilder builder(bucketize, {data, buckets});

    auto g_bucketize = ov::as_type_ptr<ov::op::v3::Bucketize>(builder.create());

    EXPECT_EQ(g_bucketize->get_output_type(), bucketize->get_output_type());
    EXPECT_EQ(g_bucketize->get_with_right_bound(), bucketize->get_with_right_bound());
}
