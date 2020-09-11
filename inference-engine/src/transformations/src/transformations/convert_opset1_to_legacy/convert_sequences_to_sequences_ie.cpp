// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_sequences_to_sequences_ie.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <ngraph_ops/lstm_sequence_ie.hpp>
#include <ngraph_ops/gru_sequence_ie.hpp>
#include <ngraph_ops/rnn_sequence_ie.hpp>

ngraph::pass::ConvertLSTMSequenceMatcher::ConvertLSTMSequenceMatcher() {
    auto lstm_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::op::v5::LSTMSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto lstm_sequence = std::dynamic_pointer_cast<ngraph::op::v5::LSTMSequence>(m.get_match_root());
        if (!lstm_sequence) {
            return false;
        }

        const auto& W = lstm_sequence->input_value(4);
        const auto& R = lstm_sequence->input_value(5);

        // Bidirectional cases are not supported
        if (lstm_sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        // for forward/reverse cases we can squeeze num_direction dimension
        auto axis_1 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset4::Squeeze>(lstm_sequence->input_value(1), axis_1);
        auto in_2 = std::make_shared<ngraph::opset4::Squeeze>(lstm_sequence->input_value(2), axis_1);
        auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::OutputVector{W, R}, 2);
        auto axis_2 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset4::Squeeze>(concat->output(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset4::Squeeze>(lstm_sequence->input_value(6), axis_2);
        auto lstm_sequence_ie = std::make_shared<ngraph::op::LSTMSequenceIE>(
                lstm_sequence->input(0).get_source_output(),  // X
                in_1,  // initial_hidden_state
                in_2,  // initial_cell_state
                lstm_sequence->input_value(3),
                in_3,  // WR
                in_4,  // B
                lstm_sequence->get_hidden_size(),
                lstm_sequence->get_direction(),
                lstm_sequence->get_activations(),
                lstm_sequence->get_activations_alpha(),
                lstm_sequence->get_activations_beta(),
                lstm_sequence->get_clip());

        auto unsqueeze_axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_1 = std::make_shared<ngraph::opset4::Unsqueeze>(lstm_sequence_ie->output(0), unsqueeze_axis);
        auto unsqueeze_2 = std::make_shared<ngraph::opset4::Unsqueeze>(lstm_sequence_ie->output(1), unsqueeze_axis);
        auto unsqueeze_3 = std::make_shared<ngraph::opset4::Unsqueeze>(lstm_sequence_ie->output(2), unsqueeze_axis);

        ngraph::copy_runtime_info(lstm_sequence, {concat, lstm_sequence_ie, in_1, in_2, in_3, in_4, unsqueeze_1,
                                                  unsqueeze_2, unsqueeze_3});
        unsqueeze_1->set_friendly_name(lstm_sequence->get_friendly_name()+".0");
        unsqueeze_2->set_friendly_name(lstm_sequence->get_friendly_name()+".1");
        unsqueeze_3->set_friendly_name(lstm_sequence->get_friendly_name()+".2");
        ngraph::replace_node(lstm_sequence, {unsqueeze_1->output(0), unsqueeze_2->output(0), unsqueeze_3->output(0)});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstm_sequence_ngraph, "ConvertLSTMSequenceToLSTMSequenceIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertGRUSequenceMatcher::ConvertGRUSequenceMatcher() {
    auto gru_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::op::v5::GRUSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto gru_sequence = std::dynamic_pointer_cast<ngraph::op::v5::GRUSequence>(m.get_match_root());
        if (!gru_sequence) {
            return false;
        }

        auto W = gru_sequence->input_value(3);
        auto R = gru_sequence->input_value(4);

        // Bidirectional cases are not supported
        if (gru_sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        // for forward/reverse cases we can squeeze num_direction dimension
        auto axis_1 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset4::Squeeze>(gru_sequence->input_value(1), axis_1);
        auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::OutputVector{W, R}, 2);
        auto axis_2 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset4::Squeeze>(concat->output(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset4::Squeeze>(gru_sequence->input_value(5), axis_2);

        auto gru_sequence_ie = std::make_shared<ngraph::op::GRUSequenceIE>(
                gru_sequence->input_value(0), // X
                in_1,  // initial_hidden_state
                gru_sequence->input_value(2),
                in_3,  // WR
                in_4,  // B
                gru_sequence->get_hidden_size(),
                gru_sequence->get_direction(),
                gru_sequence->get_activations(),
                gru_sequence->get_activations_alpha(),
                gru_sequence->get_activations_beta(),
                gru_sequence->get_clip(),
                gru_sequence->get_linear_before_reset());

        auto unsqueeze_axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_1 = std::make_shared<ngraph::opset4::Unsqueeze>(gru_sequence_ie->output(0), unsqueeze_axis);
        auto unsqueeze_2 = std::make_shared<ngraph::opset4::Unsqueeze>(gru_sequence_ie->output(1), unsqueeze_axis);

        ngraph::copy_runtime_info(gru_sequence, {concat, gru_sequence_ie, unsqueeze_1, unsqueeze_2, in_1, in_3, in_4});
        unsqueeze_1->set_friendly_name(gru_sequence->get_friendly_name()+".0");
        unsqueeze_2->set_friendly_name(gru_sequence->get_friendly_name()+".1");
        ngraph::replace_node(gru_sequence, {unsqueeze_1, unsqueeze_2});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gru_sequence_ngraph, "ConvertGRUSequenceToGRUSequenceIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertRNNSequenceMatcher::ConvertRNNSequenceMatcher() {
    auto rnn_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::op::v5::RNNSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto rnn_sequence = std::dynamic_pointer_cast<ngraph::op::v5::RNNSequence>(m.get_match_root());
        if (!rnn_sequence) {
            return false;
        }

        // Bidirectional cases are not supported
        if (rnn_sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        auto W = rnn_sequence->input_value(3);
        auto R = rnn_sequence->input_value(4);

        // for forward/reverse cases we can squeeze num_direction dimension
        auto axis_1 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset4::Squeeze>(rnn_sequence->input_value(1), axis_1);
        auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::OutputVector{W, R}, 2);
        auto axis_2 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset4::Squeeze>(concat->output(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset4::Squeeze>(rnn_sequence->input_value(5), axis_2);
        auto rnn_sequence_ie = std::make_shared<ngraph::op::RNNSequenceIE>(
                rnn_sequence->input_value(0),  // X
                in_1,  // initial_hidden_state
                rnn_sequence->input_value(2),
                in_3,  // WR
                in_4,  // B
                rnn_sequence->get_hidden_size(),
                rnn_sequence->get_direction(),
                rnn_sequence->get_activations(),
                rnn_sequence->get_activations_alpha(),
                rnn_sequence->get_activations_beta(),
                rnn_sequence->get_clip());

        auto unsqueeze_axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_1 = std::make_shared<ngraph::opset4::Unsqueeze>(rnn_sequence_ie->output(0), unsqueeze_axis);
        auto unsqueeze_2 = std::make_shared<ngraph::opset4::Unsqueeze>(rnn_sequence_ie->output(1), unsqueeze_axis);

        ngraph::copy_runtime_info(rnn_sequence, {concat, rnn_sequence_ie, in_1, in_3, in_4, unsqueeze_1,
                                                 unsqueeze_2});
        unsqueeze_1->set_friendly_name(rnn_sequence->get_friendly_name()+".0");
        unsqueeze_2->set_friendly_name(rnn_sequence->get_friendly_name()+".1");
        ngraph::replace_node(rnn_sequence, {unsqueeze_1->output(0), unsqueeze_2->output(0)});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_sequence_ngraph, "ConvertRNNSequenceToRNNSequenceIE");
    this->register_matcher(m, callback);
}