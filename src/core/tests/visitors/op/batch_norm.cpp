// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include <gtest/gtest.h>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;

TEST(attributes, batch_norm_inference_op_v5) {
    PartialShape in_shape{1, 10};
    PartialShape ch_shape{in_shape[1]};
    element::Type et = element::f32;
    double epsilon = 0.001;

    test::NodeBuilder::opset().insert<op::v5::BatchNormInference>();
    auto data_batch = make_shared<ov::op::v0::Parameter>(et, in_shape);
    auto gamma = make_shared<ov::op::v0::Parameter>(et, ch_shape);
    auto beta = make_shared<ov::op::v0::Parameter>(et, ch_shape);
    auto mean = make_shared<ov::op::v0::Parameter>(et, ch_shape);
    auto var = make_shared<ov::op::v0::Parameter>(et, ch_shape);
    auto batch_norm = make_shared<op::v5::BatchNormInference>(data_batch, gamma, beta, mean, var, epsilon);

    const auto expected_attr_count = 1;
    test::NodeBuilder builder(batch_norm, {data_batch, gamma, beta, mean, var});
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    auto g_batch_norm = ov::as_type_ptr<op::v5::BatchNormInference>(builder.create());
    EXPECT_EQ(g_batch_norm->get_eps_value(), batch_norm->get_eps_value());
}

TEST(attributes, batch_norm_inference_op_v0) {
    PartialShape in_shape{1, 10};
    PartialShape ch_shape{in_shape[1]};
    element::Type et = element::f32;
    double epsilon = 0.001;

    test::NodeBuilder::opset().insert<op::v0::BatchNormInference>();
    auto data_batch = make_shared<ov::op::v0::Parameter>(et, in_shape);
    auto gamma = make_shared<ov::op::v0::Parameter>(et, ch_shape);
    auto beta = make_shared<ov::op::v0::Parameter>(et, ch_shape);
    auto mean = make_shared<ov::op::v0::Parameter>(et, ch_shape);
    auto var = make_shared<ov::op::v0::Parameter>(et, ch_shape);
    auto batch_norm = make_shared<op::v0::BatchNormInference>(data_batch, gamma, beta, mean, var, epsilon);

    const auto expected_attr_count = 1;
    test::NodeBuilder builder(batch_norm, {gamma, beta, data_batch, mean, var});
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    auto g_batch_norm = ov::as_type_ptr<op::v0::BatchNormInference>(builder.create());
    EXPECT_EQ(g_batch_norm->get_eps_value(), batch_norm->get_eps_value());
}
