// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/op_conversions/rnn_cell_fusion.hpp>
#include <transformations/op_conversions/gru_cell_fusion.hpp>
#include <transformations/op_conversions/lstm_cell_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, RNNCellFusion) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const auto hidden_size = 3;
    {
        const auto X = std::make_shared<opset8::Parameter>(element::f32, PartialShape{batch_size, input_size});
        const auto H_t = std::make_shared<opset8::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
        const auto W = std::make_shared<opset8::Parameter>(element::f32, PartialShape{hidden_size, input_size});
        const auto R = std::make_shared<opset8::Parameter>(element::f32, PartialShape{hidden_size, hidden_size});
        const auto B = std::make_shared<opset8::Parameter>(element::f32, PartialShape{hidden_size});

        auto Xt_W = std::make_shared<opset8::MatMul>(X, W, false, true);
        auto Ht_R = std::make_shared<opset8::MatMul>(H_t, R, false, true);
        auto add = std::make_shared<opset8::Add>(Ht_R, B);
        auto i_t = std::make_shared<opset8::Add>(Xt_W, add);

        auto activation = std::make_shared<opset8::Sigmoid>(i_t);
        auto result = std::make_shared<opset8::Result>(activation);
        function = std::make_shared<Model>(ResultVector{result}, ParameterVector{X, H_t, W, R, B});
        manager.register_pass<pass::RNNCellFusion>();
    }

    {
        const auto X = std::make_shared<opset8::Parameter>(element::f32, PartialShape{batch_size, input_size});
        const auto H_t = std::make_shared<opset8::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
        const auto W = std::make_shared<opset8::Parameter>(element::f32, PartialShape{hidden_size, input_size});
        const auto R = std::make_shared<opset8::Parameter>(element::f32, PartialShape{hidden_size, hidden_size});
        const auto B = std::make_shared<opset8::Parameter>(element::f32, PartialShape{hidden_size});

        auto rnn_cell = std::make_shared<opset8::RNNCell>(X, H_t, W, R, B, hidden_size, std::vector<std::string>{"sigmoid"});
        auto result = std::make_shared<opset8::Result>(rnn_cell);
        function_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{X, H_t, W, R, B});
    }
}
