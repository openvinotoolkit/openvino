// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_lstm_cell_to_lstm_cell_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/lstm_cell_ie.hpp>

void ngraph::pass::ConvertLSTMCellToLSTMCellIE::convert_lstm_cell() {
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

        auto W = std::dynamic_pointer_cast<ngraph::opset1::Constant> (lstm_cell->input(3).get_source_output().get_node_shared_ptr());
        if (!W) {
            return false;
        }

        auto R = std::dynamic_pointer_cast<ngraph::opset1::Constant> (lstm_cell->input(4).get_source_output().get_node_shared_ptr());
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
                                                                      lstm_cell->get_clip(),
                                                                      lstm_cell->get_output_shape(0),
                                                                      lstm_cell->get_output_shape(1));

        lstm_cell_ie->set_friendly_name(lstm_cell->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), lstm_cell_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstm_cell_ngraph, "CPUFusion.ConvertLSTMCellToLSTMCellIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}