// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <ngraph/ops.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/function.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_cells_to_cells_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <legacy/ngraph_ops/gru_cell_ie.hpp>
#include <legacy/ngraph_ops/rnn_cell_ie.hpp>
#include <legacy/ngraph_ops/lstm_cell_ie.hpp>
#include <ngraph/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, GRUCellConversionTest) {
    std::shared_ptr<ngraph::opset3::GRUCell> cell;

    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    {
        const auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, input_size});
        const auto W =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, hidden_size});
        const auto B = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size});
        cell = std::make_shared<ngraph::opset3::GRUCell>(X, H_t, W, R, B, hidden_size);
        cell->set_friendly_name("test_cell");

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell}, ngraph::ParameterVector{X, H_t});
        manager.register_pass<ngraph::pass::ConvertGRUCellMatcher>();
    }

    {
        const auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, input_size});
        const auto W =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, hidden_size});
        const auto B = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size});
        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector({W, R}), 1);
        auto cell_ie = std::make_shared<ngraph::op::GRUCellIE>(X, H_t, concat, B,
                                                               cell->get_hidden_size(),
                                                               cell->get_activations(),
                                                               cell->get_activations_alpha(),
                                                               cell->get_activations_beta(),
                                                               cell->get_clip(),
                                                               cell->get_linear_before_reset());
        cell_ie->set_friendly_name("test_cell");

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell_ie}, ngraph::ParameterVector{X, H_t});
    }
}

TEST_F(TransformationTestsF, RNNCellConversionTest) {
    const size_t hidden_size = 3;
    std::shared_ptr<ngraph::opset3::RNNCell> cell;

    {
        auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto H = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto W = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto R = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto B = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3});

        cell = std::make_shared<ngraph::opset3::RNNCell>(X, H, W, R, B, hidden_size);
        cell->set_friendly_name("test_cell");

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell}, ngraph::ParameterVector{X, H});
        manager.register_pass<ngraph::pass::ConvertRNNCellMatcher>();
    }

    {
        auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto H = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto W = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto R = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto B = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3});
        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector({W, R}), 1);
        auto cell_ie = std::make_shared<ngraph::op::RNNCellIE>(X, H, concat, B,
                                                               cell->get_hidden_size(),
                                                               cell->get_activations(),
                                                               cell->get_activations_alpha(),
                                                               cell->get_activations_beta(),
                                                               cell->get_clip());

        cell_ie->set_friendly_name("test_cell");
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell_ie}, ngraph::ParameterVector{X, H});
    }
}

TEST_F(TransformationTestsF, LSTMCellConversionTest_opset3) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    std::shared_ptr<ngraph::opset3::LSTMCell> cell;
    {
        const auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, input_size});
        const auto W =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, hidden_size});
        const auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, hidden_size});
        const auto B = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size});

        cell = std::make_shared<ngraph::opset3::LSTMCell>(X, H_t, C_t, W, R, B, hidden_size);
        cell->set_friendly_name("test_cell");

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell}, ngraph::ParameterVector{X, H_t, C_t});
        manager.register_pass<ngraph::pass::ConvertLSTMCellMatcher>();
    }

    {
        const auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, input_size});
        const auto W =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, hidden_size});
        const auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, hidden_size});
        const auto B = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{gates_count * hidden_size});

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector({W, R}), 1);
        auto cell_ie = std::make_shared<ngraph::op::LSTMCellIE>(X, H_t, C_t, concat, B,
                                                                cell->get_hidden_size(),
                                                                cell->get_activations(),
                                                                cell->get_activations_alpha(),
                                                                cell->get_activations_beta(),
                                                                cell->get_clip());
        cell_ie->set_friendly_name("test_cell");

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell_ie}, ngraph::ParameterVector{X, H_t, C_t});
    }
}

TEST_F(TransformationTestsF, LSTMCellConversionTest_opset4) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    std::shared_ptr<ngraph::opset4::LSTMCell> cell;
    {
        const auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, input_size});
        const auto W =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, hidden_size});
        const auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, hidden_size});
        const auto B = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{gates_count * hidden_size});

        cell = std::make_shared<ngraph::opset4::LSTMCell>(X, H_t, C_t, W, R, B, hidden_size);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell}, ngraph::ParameterVector{X, H_t, C_t});
        manager.register_pass<ngraph::pass::ConvertLSTMCellMatcher>();
    }

    {
        const auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, input_size});
        const auto W =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, hidden_size});
        const auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, hidden_size});
        const auto B = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{gates_count * hidden_size});

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector({W, R}), 1);
        auto cell_ie = std::make_shared<ngraph::op::LSTMCellIE>(X, H_t, C_t, concat, B,
                                                                cell->get_hidden_size(),
                                                                cell->get_activations(),
                                                                cell->get_activations_alpha(),
                                                                cell->get_activations_beta(),
                                                                cell->get_clip());
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell_ie}, ngraph::ParameterVector{X, H_t, C_t});
    }
}
