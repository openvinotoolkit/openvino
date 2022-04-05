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

TEST(attributes, gru_cell_op) {
    NodeBuilder::get_ops().register_factory<opset5::GRUCell>();
    auto X = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{9, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{9, 3});
    const auto initial_hidden_state = make_shared<op::Parameter>(element::f32, Shape{2, 3});

    const auto hidden_size = 3;
    const std::vector<std::string> activations = {"tanh", "sigmoid"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    const float clip = 0.5f;
    const auto gru_cell = make_shared<opset5::GRUCell>(X,
                                                       initial_hidden_state,
                                                       W,
                                                       R,
                                                       hidden_size,
                                                       activations,
                                                       activations_alpha,
                                                       activations_beta,
                                                       clip,
                                                       false);
    NodeBuilder builder(gru_cell);
    auto g_gru_cell = ov::as_type_ptr<opset5::GRUCell>(builder.create());

    EXPECT_EQ(g_gru_cell->get_hidden_size(), gru_cell->get_hidden_size());
    EXPECT_EQ(g_gru_cell->get_activations(), gru_cell->get_activations());
    EXPECT_EQ(g_gru_cell->get_activations_alpha(), gru_cell->get_activations_alpha());
    EXPECT_EQ(g_gru_cell->get_activations_beta(), gru_cell->get_activations_beta());
    EXPECT_EQ(g_gru_cell->get_clip(), gru_cell->get_clip());
    EXPECT_EQ(g_gru_cell->get_linear_before_reset(), gru_cell->get_linear_before_reset());
}
