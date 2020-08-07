// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset4_to_opset3/convert_sequences_to_sequences_ie.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <ngraph_ops/lstm_sequence_ie.hpp>
#include <ngraph_ops/gru_sequence_ie.hpp>
#include <ngraph_ops/rnn_sequence_ie.hpp>

ngraph::pass::ConvertLSTMSequenceMatcher::ConvertLSTMSequenceMatcher() {
    auto lstm_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::opset4::LSTMSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto lstm_sequence = std::dynamic_pointer_cast<ngraph::opset4::LSTMSequence>(m.get_match_root());
        if (!lstm_sequence) {
            return false;
        }

        auto W = std::dynamic_pointer_cast<ngraph::opset4::Constant>(
                lstm_sequence->input_value(4).get_node_shared_ptr());
        if (!W) {
            return false;
        }

        auto R = std::dynamic_pointer_cast<ngraph::opset4::Constant>(
                lstm_sequence->input_value(5).get_node_shared_ptr());
        if (!R) {
            return false;
        }

        auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::NodeVector({W, R}), 2);
        auto lstm_sequence_ie = std::make_shared<ngraph::op::LSTMSequenceIE>(
                lstm_sequence->input(0).get_source_output(),  // X
                lstm_sequence->input(1).get_source_output(),  // initial_hidden_state
                lstm_sequence->input(2).get_source_output(),  // initial_cell_state
                lstm_sequence->input(3).get_source_output(),  // sequence_lengths
                concat->output(0),                           // WR
                lstm_sequence->input(6).get_source_output(),  // B
                lstm_sequence->get_hidden_size(),
                lstm_sequence->get_direction(),
                lstm_sequence->get_activations(),
                lstm_sequence->get_activations_alpha(),
                lstm_sequence->get_activations_beta(),
                lstm_sequence->get_clip_threshold());

        lstm_sequence_ie->set_friendly_name(lstm_sequence->get_friendly_name());
        ngraph::copy_runtime_info(lstm_sequence, {concat, lstm_sequence_ie});
        ngraph::replace_node(m.get_match_root(), lstm_sequence_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstm_sequence_ngraph, "ConvertLSTMSequenceToLSTMSequenceIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertGRUSequenceMatcher::ConvertGRUSequenceMatcher() {
    auto gru_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::opset4::GRUSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto gru_sequence = std::dynamic_pointer_cast<ngraph::opset4::GRUSequence>(m.get_match_root());
        if (!gru_sequence) {
            return false;
        }

        auto W = std::dynamic_pointer_cast<ngraph::opset4::Constant>(
                gru_sequence->input_value(3).get_node_shared_ptr());
        if (!W) {
            return false;
        }

        auto R = std::dynamic_pointer_cast<ngraph::opset4::Constant>(
                gru_sequence->input_value(4).get_node_shared_ptr());
        if (!R) {
            return false;
        }

        auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::NodeVector({W, R}), 2);
        auto gru_sequence_ie = std::make_shared<ngraph::op::GRUSequenceIE>(
                gru_sequence->input(0).get_source_output(),  // X
                gru_sequence->input(1).get_source_output(),  // initial_hidden_state
                gru_sequence->input(2).get_source_output(),  // sequence_lengths
                concat->output(0),                          // WR
                gru_sequence->input(5).get_source_output(),  // B
                gru_sequence->get_hidden_size(),
                gru_sequence->get_direction(),
                gru_sequence->get_activations(),
                gru_sequence->get_activations_alpha(),
                gru_sequence->get_activations_beta(),
                gru_sequence->get_clip(),
                gru_sequence->get_linear_before_reset());

        gru_sequence_ie->set_friendly_name(gru_sequence->get_friendly_name());
        ngraph::copy_runtime_info(gru_sequence, {concat, gru_sequence_ie});
        ngraph::replace_node(m.get_match_root(), gru_sequence_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gru_sequence_ngraph, "ConvertGRUSequenceToGRUSequenceIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertRNNSequenceMatcher::ConvertRNNSequenceMatcher() {
    auto rnn_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::opset4::RNNSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto rnn_sequence = std::dynamic_pointer_cast<ngraph::opset4::RNNSequence>(m.get_match_root());
        if (!rnn_sequence) {
            return false;
        }

        auto W = std::dynamic_pointer_cast<ngraph::opset4::Constant>(
                rnn_sequence->input_value(3).get_node_shared_ptr());
        if (!W) {
            return false;
        }

        auto R = std::dynamic_pointer_cast<ngraph::opset4::Constant>(
                rnn_sequence->input_value(4).get_node_shared_ptr());
        if (!R) {
            return false;
        }

        auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::NodeVector({W, R}), 2);
        auto rnn_sequence_ie = std::make_shared<ngraph::op::RNNSequenceIE>(
                rnn_sequence->input(0).get_source_output(),  // X
                rnn_sequence->input(1).get_source_output(),  // initial_hidden_state
                rnn_sequence->input(2).get_source_output(),  // sequence_lengths
                concat->output(0),                          // WR
                rnn_sequence->input(5).get_source_output(),  // B
                rnn_sequence->get_hidden_size(),
                rnn_sequence->get_direction(),
                rnn_sequence->get_activations(),
                rnn_sequence->get_activations_alpha(),
                rnn_sequence->get_activations_beta(),
                rnn_sequence->get_clip());

        rnn_sequence_ie->set_friendly_name(rnn_sequence->get_friendly_name());
        ngraph::copy_runtime_info(rnn_sequence, {concat, rnn_sequence_ie});
        ngraph::replace_node(m.get_match_root(), rnn_sequence_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_sequence_ngraph, "ConvertRNNSequenceToRNNSequenceIE");
    this->register_matcher(m, callback);
}