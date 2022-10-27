// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, batch_norm_inference_op_v5) {
    PartialShape in_shape{1, 10};
    PartialShape ch_shape{in_shape[1]};
    element::Type et = element::f32;
    double epsilon = 0.001;

    NodeBuilder::get_ops().register_factory<op::v5::BatchNormInference>();
    auto data_batch = make_shared<op::Parameter>(et, in_shape);
    auto gamma = make_shared<op::Parameter>(et, ch_shape);
    auto beta = make_shared<op::Parameter>(et, ch_shape);
    auto mean = make_shared<op::Parameter>(et, ch_shape);
    auto var = make_shared<op::Parameter>(et, ch_shape);
    auto batch_norm = make_shared<op::v5::BatchNormInference>(data_batch, gamma, beta, mean, var, epsilon);

    const auto expected_attr_count = 1;
    NodeBuilder builder(batch_norm, {data_batch, gamma, beta, mean, var});
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    auto g_batch_norm = ov::as_type_ptr<op::v5::BatchNormInference>(builder.create());
    EXPECT_EQ(g_batch_norm->get_eps_value(), batch_norm->get_eps_value());
}

TEST(attributes, batch_norm_inference_op_v0) {
    PartialShape in_shape{1, 10};
    PartialShape ch_shape{in_shape[1]};
    element::Type et = element::f32;
    double epsilon = 0.001;

    NodeBuilder::get_ops().register_factory<op::v0::BatchNormInference>();
    auto data_batch = make_shared<op::Parameter>(et, in_shape);
    auto gamma = make_shared<op::Parameter>(et, ch_shape);
    auto beta = make_shared<op::Parameter>(et, ch_shape);
    auto mean = make_shared<op::Parameter>(et, ch_shape);
    auto var = make_shared<op::Parameter>(et, ch_shape);
    auto batch_norm = make_shared<op::v0::BatchNormInference>(data_batch, gamma, beta, mean, var, epsilon);

    const auto expected_attr_count = 1;
    NodeBuilder builder(batch_norm, {gamma, beta, data_batch, mean, var});
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    auto g_batch_norm = ov::as_type_ptr<op::v0::BatchNormInference>(builder.create());
    EXPECT_EQ(g_batch_norm->get_eps_value(), batch_norm->get_eps_value());
}
