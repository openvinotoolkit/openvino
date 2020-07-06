// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_cells_to_cells_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

#include <ngraph_ops/lstm_cell_ie.hpp>
#include <ngraph_ops/gru_cell_ie.hpp>
#include <ngraph_ops/rnn_cell_ie.hpp>

ngraph::pass::ConvertLSTMCellMatcher::ConvertLSTMCellMatcher() {
    // placeholders
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});  // X
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1});  // initial_hidden_state
    auto input_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1});  // initial_cell_state
    auto input_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{4, 1});  // W
    auto input_4 = std::make_shared<pattern::op::Label>(element::f32, Shape{4, 1});  // R
    auto input_5 = std::make_shared<pattern::op::Label>(element::f32, Shape{4});     // B

    auto lstm_cell_ngraph = std::make_shared<ngraph::opset1::LSTMCell>(input_0, input_1, input_2, input_3, input_4, input_5, 1);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto lstm_cell = std::dynamic_pointer_cast<ngraph::opset1::LSTMCell> (m.get_match_root());
        if (!lstm_cell) {
            return false;
        }

        auto W = std::dynamic_pointer_cast<ngraph::opset1::Constant> (lstm_cell->input_value(3).get_node_shared_ptr());
        if (!W) {
            return false;
        }

        auto R = std::dynamic_pointer_cast<ngraph::opset1::Constant> (lstm_cell->input_value(4).get_node_shared_ptr());
        if (!R) {
            return false;
        }

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector({W, R}), 1);
        auto lstm_cell_ie = std::make_shared<ngraph::op::LSTMCellIE> (lstm_cell->input(0).get_source_output(),  // X
                                                                      lstm_cell->input(1).get_source_output(),  // initial_hidden_state
                                                                      lstm_cell->input(2).get_source_output(),  // initial_cell_state
                                                                      concat->output(0),                       // WR
                                                                      lstm_cell->input(5).get_source_output(),  // B
                                                                      lstm_cell->get_hidden_size(),
                                                                      lstm_cell->get_activations(),
                                                                      lstm_cell->get_activations_alpha(),
                                                                      lstm_cell->get_activations_beta(),
                                                                      lstm_cell->get_clip());

        lstm_cell_ie->set_friendly_name(lstm_cell->get_friendly_name());
        ngraph::copy_runtime_info(lstm_cell, {concat, lstm_cell_ie});
        ngraph::replace_node(m.get_match_root(), lstm_cell_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstm_cell_ngraph, "ConvertLSTMCellToLSTMCellIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertGRUCellMatcher::ConvertGRUCellMatcher() {
    // placeholders
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1});  // X
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1});  // initial_hidden_state
    auto input_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 1});  // W
    auto input_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 1});  // R
    auto input_4 = std::make_shared<pattern::op::Label>(element::f32, Shape{3});     // B

    auto gru_cell_ngraph = std::make_shared<ngraph::opset3::GRUCell>(input_0, input_1, input_2, input_3, input_4, 1);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto gru_cell = std::dynamic_pointer_cast<ngraph::opset3::GRUCell> (m.get_match_root());
        if (!gru_cell) {
            return false;
        }

        auto W = std::dynamic_pointer_cast<ngraph::opset1::Constant> (gru_cell->input_value(2).get_node_shared_ptr());
        if (!W) {
            return false;
        }

        auto R = std::dynamic_pointer_cast<ngraph::opset1::Constant> (gru_cell->input_value(3).get_node_shared_ptr());
        if (!R) {
            return false;
        }

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector({W, R}), 1);
        auto gru_cell_ie = std::make_shared<ngraph::op::GRUCellIE>(gru_cell->input(0).get_source_output(),  // X
                                                                   gru_cell->input(1).get_source_output(),  // initial_hidden_state
                                                                   concat->output(0),                       // WR
                                                                   gru_cell->input(4).get_source_output(),  // B
                                                                   gru_cell->get_hidden_size(),
                                                                   gru_cell->get_activations(),
                                                                   gru_cell->get_activations_alpha(),
                                                                   gru_cell->get_activations_beta(),
                                                                   gru_cell->get_clip(),
                                                                   gru_cell->get_linear_before_reset());

        gru_cell_ie->set_friendly_name(gru_cell->get_friendly_name());
        ngraph::copy_runtime_info(gru_cell, {concat, gru_cell_ie});
        ngraph::replace_node(m.get_match_root(), gru_cell_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gru_cell_ngraph, "ConvertGRUCellToGRUCellIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertRNNCellMatcher::ConvertRNNCellMatcher() {
    // placeholders
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1});  // X
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1});  // initial_hidden_state
    auto input_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1});  // W
    auto input_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1});  // R
    auto input_4 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});     // B

    auto rnn_cell_ngraph = std::make_shared<ngraph::opset3::RNNCell>(input_0, input_1, input_2, input_3, input_4, 1);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto rnn_cell = std::dynamic_pointer_cast<ngraph::opset3::RNNCell> (m.get_match_root());
        if (!rnn_cell) {
            return false;
        }

        auto W = std::dynamic_pointer_cast<ngraph::opset1::Constant> (rnn_cell->input_value(2).get_node_shared_ptr());
        if (!W) {
            return false;
        }

        auto R = std::dynamic_pointer_cast<ngraph::opset1::Constant> (rnn_cell->input_value(3).get_node_shared_ptr());
        if (!R) {
            return false;
        }

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector({W, R}), 1);
        auto rnn_cell_ie = std::make_shared<ngraph::op::RNNCellIE> (rnn_cell->input(0).get_source_output(),  // X
                                                                    rnn_cell->input(1).get_source_output(),  // initial_hidden_state
                                                                    concat->output(0),                      // WR
                                                                    rnn_cell->input(4).get_source_output(),  // B
                                                                    rnn_cell->get_hidden_size(),
                                                                    rnn_cell->get_activations(),
                                                                    rnn_cell->get_activations_alpha(),
                                                                    rnn_cell->get_activations_beta(),
                                                                    rnn_cell->get_clip());

        rnn_cell_ie->set_friendly_name(rnn_cell->get_friendly_name());
        ngraph::copy_runtime_info(rnn_cell, {concat, rnn_cell_ie});
        ngraph::replace_node(m.get_match_root(), rnn_cell_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_cell_ngraph, "ConvertRNNCellToRNNCellIE");
    this->register_matcher(m, callback);
}