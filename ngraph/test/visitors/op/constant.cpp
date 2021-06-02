// Copyright (C) 2018-2021 Intel Corporation
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

TEST(attributes, constant_op)
{
    vector<float> data{5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f};
    auto k = make_shared<op::v0::Constant>(element::f32, Shape{2, 3}, data);
    NodeBuilder builder(k);
    auto g_k = as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<float> g_data = g_k->get_vector<float>();
    EXPECT_EQ(data, g_data);
}

TEST(attributes, constant_op_different_elements)
{
    vector<int64_t> data{5, 4, 3, 2, 1, 0};
    auto k = make_shared<op::v0::Constant>(element::i64, Shape{2, 3}, data);
    NodeBuilder builder(k);
    auto g_k = as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<int64_t> g_data = g_k->get_vector<int64_t>();
    EXPECT_EQ(data, g_data);
    ASSERT_FALSE(g_k->get_all_data_elements_bitwise_identical());
}

TEST(attributes, constant_op_identical_elements)
{
    vector<int64_t> data{5, 5, 5, 5, 5, 5};
    auto k = make_shared<op::v0::Constant>(element::i64, Shape{2, 3}, data);
    NodeBuilder builder(k);
    auto g_k = as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<int64_t> g_data = g_k->get_vector<int64_t>();
    EXPECT_EQ(data, g_data);
    ASSERT_TRUE(g_k->get_all_data_elements_bitwise_identical());
}
