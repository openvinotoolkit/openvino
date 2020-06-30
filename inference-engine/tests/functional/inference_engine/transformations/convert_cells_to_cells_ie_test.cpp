// Copyright (C) 2020 Intel Corporation
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
#include <ngraph/function.hpp>
#include <transformations/convert_opset1_to_legacy/convert_cells_to_cells_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph_ops/gru_cell_ie.hpp>
#include <ngraph_ops/rnn_cell_ie.hpp>
#include <ngraph_ops/lstm_cell_ie.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, GRUCellConversionTest) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
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

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell}, ngraph::ParameterVector{X, H_t});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertCellsToCellsIE().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
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

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell_ie}, ngraph::ParameterVector{X, H_t});
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto cell_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(cell_node->get_friendly_name() == "test_cell") << "Transformation ConvertGRUCellToGRUCellIE should keep output names.\n";
}

TEST(TransformationTests, RNNCellConversionTest) {
    const size_t hidden_size = 3;
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    std::shared_ptr<ngraph::opset3::RNNCell> cell;

    {
        auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto H = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto W = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto R = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto B = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3});

        cell = std::make_shared<ngraph::opset3::RNNCell>(X, H, W, R, B, hidden_size);
        cell->set_friendly_name("test_cell");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell}, ngraph::ParameterVector{X, H});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertCellsToCellsIE().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
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
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell_ie}, ngraph::ParameterVector{X, H});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto cell_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(cell_node->get_friendly_name() == "test_cell") << "Transformation ConvertRNNCellToRNNCellIE should keep output names.\n";
}

TEST(TransformationTests, LSTMCellConversionTest) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
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

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell}, ngraph::ParameterVector{X, H_t, C_t});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertCellsToCellsIE().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
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

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cell_ie}, ngraph::ParameterVector{X, H_t, C_t});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto cell_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(cell_node->get_friendly_name() == "test_cell") << "Transformation ConvertLSTMCellToLSTMCellIE should keep output names.\n";
}