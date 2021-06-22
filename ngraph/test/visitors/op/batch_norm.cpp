// Copyright (C) 2021 Intel Corporation
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

template<class T>
class BatchNormAttrTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(BatchNormAttrTest);

TYPED_TEST_P(BatchNormAttrTest, batch_norm_inference_op)
{
    PartialShape in_shape{1, 10};
    PartialShape ch_shape{in_shape[1]};
    element::Type et = element::f32;
    double epsilon = 0.001;

    NodeBuilder::get_ops().register_factory<TypeParam>();
    auto data_batch = make_shared<op::Parameter>(et, in_shape);
    auto gamma = make_shared<op::Parameter>(et, ch_shape);
    auto beta = make_shared<op::Parameter>(et, ch_shape);
    auto mean = make_shared<op::Parameter>(et, ch_shape);
    auto var = make_shared<op::Parameter>(et, ch_shape);
    auto batch_norm = make_shared<TypeParam>(data_batch, gamma, beta, mean, var, epsilon);

    const auto expected_attr_count = 1;
    NodeBuilder builder(batch_norm);
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    auto g_batch_norm = as_type_ptr<TypeParam>(builder.create());
    EXPECT_EQ(g_batch_norm->get_eps_value(), batch_norm->get_eps_value());
}

REGISTER_TYPED_TEST_CASE_P(
    BatchNormAttrTest,
    batch_norm_inference_op);

using Types = ::testing::Types<op::v0::BatchNormInference, op::v5::BatchNormInference>;

INSTANTIATE_TYPED_TEST_CASE_P(attributes, BatchNormAttrTest, Types);
