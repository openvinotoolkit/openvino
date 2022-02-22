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

TEST(attributes, rnn_cell_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<opset1::RNNCell>();
    auto X = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;
    auto activations = std::vector<std::string>{"sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    float clip = 1.0;

    auto rnn_cell =
        make_shared<opset1::RNNCell>(X, H, W, R, hidden_size, activations, activations_alpha, activations_beta, clip);

    NodeBuilder builder(rnn_cell);
    auto g_rnn_cell = ov::as_type_ptr<opset1::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
}

TEST(attributes, rnn_cell_op_default_attributes) {
    NodeBuilder::get_ops().register_factory<opset1::RNNCell>();
    auto X = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;

    auto rnn_cell = make_shared<opset1::RNNCell>(X, H, W, R, hidden_size);

    NodeBuilder builder(rnn_cell);
    auto g_rnn_cell = ov::as_type_ptr<opset1::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
}
