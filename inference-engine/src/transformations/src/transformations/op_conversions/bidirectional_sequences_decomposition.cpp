// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::BidirectionalLSTMSequenceDecomposition, "BidirectionalLSTMSequenceDecomposition", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::BidirectionalGRUSequenceDecomposition, "BidirectionalGRUSequenceDecomposition", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::BidirectionalRNNSequenceDecomposition, "BidirectionalRNNSequenceDecomposition", 0);

ngraph::pass::BidirectionalLSTMSequenceDecomposition::BidirectionalLSTMSequenceDecomposition() {
    auto lstm_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::opset5::LSTMSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto lstm_sequence = std::dynamic_pointer_cast<ngraph::opset5::LSTMSequence>(m.get_match_root());
        if (!lstm_sequence) {
            return false;
        }

        if (lstm_sequence->get_direction() != ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        auto axis_0 = ngraph::opset4::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = ngraph::opset4::Constant::create(element::i64, Shape{}, {1});
        auto H = std::make_shared<opset4::Split>(lstm_sequence->input_value(1), axis_1, 2);
        auto C = std::make_shared<opset4::Split>(lstm_sequence->input_value(2), axis_1, 2);
        auto W = std::make_shared<opset4::Split>(lstm_sequence->input_value(4), axis_0, 2);
        auto R = std::make_shared<opset4::Split>(lstm_sequence->input_value(5), axis_0, 2);
        auto B = std::make_shared<opset4::Split>(lstm_sequence->input_value(6), axis_0, 2);
        auto lstm_sequence_forward = std::make_shared<ngraph::op::v5::LSTMSequence>(
                lstm_sequence->input_value(0),
                H->output(0),
                C->output(0),
                lstm_sequence->input_value(3),
                W->output(0),
                R->output(0),
                B->output(0),
                lstm_sequence->get_hidden_size(),
                ngraph::op::RecurrentSequenceDirection::FORWARD,
                lstm_sequence->get_activations_alpha(),
                lstm_sequence->get_activations_beta(),
                lstm_sequence->get_activations(),
                lstm_sequence->get_clip());

        auto lstm_sequence_reverse = std::make_shared<ngraph::opset5::LSTMSequence>(
                lstm_sequence->input_value(0),
                H->output(1),
                C->output(1),
                lstm_sequence->input_value(3),
                W->output(1),
                R->output(1),
                B->output(1),
                lstm_sequence->get_hidden_size(),
                ngraph::op::RecurrentSequenceDirection::REVERSE,
                lstm_sequence->get_activations_alpha(),
                lstm_sequence->get_activations_beta(),
                lstm_sequence->get_activations(),
                lstm_sequence->get_clip());

        auto concat_0 = std::make_shared<opset5::Concat>(OutputVector{lstm_sequence_forward->output(0),
                                                          lstm_sequence_reverse->output(0)}, 1);
        auto concat_1 = std::make_shared<opset5::Concat>(OutputVector{lstm_sequence_forward->output(1),
                                                          lstm_sequence_reverse->output(1)}, 1);
        auto concat_2 = std::make_shared<opset5::Concat>(OutputVector{lstm_sequence_forward->output(2),
                                                          lstm_sequence_reverse->output(2)}, 1);
        ngraph::copy_runtime_info(lstm_sequence, {H, C, W, R, B, lstm_sequence_forward, lstm_sequence_reverse,
                                                  concat_0, concat_1, concat_2});
        concat_0->set_friendly_name(lstm_sequence->get_friendly_name()+".0");
        concat_1->set_friendly_name(lstm_sequence->get_friendly_name()+".1");
        concat_2->set_friendly_name(lstm_sequence->get_friendly_name()+".2");
        ngraph::replace_node(lstm_sequence, {concat_0->output(0), concat_1->output(0), concat_2->output(0)});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstm_sequence_ngraph, "BidirectionalLSTMSequenceDecomposition");
    this->register_matcher(m, callback);
}

ngraph::pass::BidirectionalGRUSequenceDecomposition::BidirectionalGRUSequenceDecomposition() {
    auto gru_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::opset5::GRUSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto gru_sequence = std::dynamic_pointer_cast<ngraph::opset5::GRUSequence>(m.get_match_root());
        if (!gru_sequence) {
            return false;
        }

        if (gru_sequence->get_direction() != ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        auto axis_0 = ngraph::opset4::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = ngraph::opset4::Constant::create(element::i64, Shape{}, {1});
        auto H = std::make_shared<opset4::Split>(gru_sequence->input_value(1), axis_1, 2);
        auto W = std::make_shared<opset4::Split>(gru_sequence->input_value(3), axis_0, 2);
        auto R = std::make_shared<opset4::Split>(gru_sequence->input_value(4), axis_0, 2);
        auto B = std::make_shared<opset4::Split>(gru_sequence->input_value(5), axis_0, 2);
        auto gru_sequence_forward = std::make_shared<ngraph::op::v5::GRUSequence>(
                gru_sequence->input_value(0),
                H->output(0),
                gru_sequence->input_value(2),
                W->output(0),
                R->output(0),
                B->output(0),
                gru_sequence->get_hidden_size(),
                ngraph::op::RecurrentSequenceDirection::FORWARD,
                gru_sequence->get_activations(),
                gru_sequence->get_activations_alpha(),
                gru_sequence->get_activations_beta(),
                gru_sequence->get_clip(),
                gru_sequence->get_linear_before_reset());

        auto gru_sequence_reverse = std::make_shared<ngraph::opset5::GRUSequence>(
                gru_sequence->input_value(0),
                H->output(1),
                gru_sequence->input_value(2),
                W->output(1),
                R->output(1),
                B->output(1),
                gru_sequence->get_hidden_size(),
                ngraph::op::RecurrentSequenceDirection::REVERSE,
                gru_sequence->get_activations(),
                gru_sequence->get_activations_alpha(),
                gru_sequence->get_activations_beta(),
                gru_sequence->get_clip(),
                gru_sequence->get_linear_before_reset());

        auto concat_0 = std::make_shared<opset5::Concat>(OutputVector{gru_sequence_forward->output(0),
                                                                      gru_sequence_reverse->output(0)}, 1);
        auto concat_1 = std::make_shared<opset5::Concat>(OutputVector{gru_sequence_forward->output(1),
                                                                      gru_sequence_reverse->output(1)}, 1);
        ngraph::copy_runtime_info(gru_sequence, {H, W, R, B, gru_sequence_forward, gru_sequence_reverse,
                                                  concat_0, concat_1});
        concat_0->set_friendly_name(gru_sequence->get_friendly_name()+".0");
        concat_1->set_friendly_name(gru_sequence->get_friendly_name()+".1");
        ngraph::replace_node(gru_sequence, {concat_0->output(0), concat_1->output(0)});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gru_sequence_ngraph, "BidirectionalGRUSequenceDecomposition");
    this->register_matcher(m, callback);
}

ngraph::pass::BidirectionalRNNSequenceDecomposition::BidirectionalRNNSequenceDecomposition() {
    auto rnn_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::opset5::RNNSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto rnn_sequence = std::dynamic_pointer_cast<ngraph::opset5::RNNSequence>(m.get_match_root());
        if (!rnn_sequence) {
            return false;
        }

        if (rnn_sequence->get_direction() != ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        auto axis_0 = ngraph::opset4::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = ngraph::opset4::Constant::create(element::i64, Shape{}, {1});
        auto H = std::make_shared<opset4::Split>(rnn_sequence->input_value(1), axis_1, 2);
        auto W = std::make_shared<opset4::Split>(rnn_sequence->input_value(3), axis_0, 2);
        auto R = std::make_shared<opset4::Split>(rnn_sequence->input_value(4), axis_0, 2);
        auto B = std::make_shared<opset4::Split>(rnn_sequence->input_value(5), axis_0, 2);
        auto rnn_sequence_forward = std::make_shared<ngraph::op::v5::RNNSequence>(
                rnn_sequence->input_value(0),
                H->output(0),
                rnn_sequence->input_value(2),
                W->output(0),
                R->output(0),
                B->output(0),
                rnn_sequence->get_hidden_size(),
                ngraph::op::RecurrentSequenceDirection::FORWARD,
                rnn_sequence->get_activations(),
                rnn_sequence->get_activations_alpha(),
                rnn_sequence->get_activations_beta(),
                rnn_sequence->get_clip());

        auto rnn_sequence_reverse = std::make_shared<ngraph::opset5::RNNSequence>(
                rnn_sequence->input_value(0),
                H->output(1),
                rnn_sequence->input_value(2),
                W->output(1),
                R->output(1),
                B->output(1),
                rnn_sequence->get_hidden_size(),
                ngraph::op::RecurrentSequenceDirection::REVERSE,
                rnn_sequence->get_activations(),
                rnn_sequence->get_activations_alpha(),
                rnn_sequence->get_activations_beta(),
                rnn_sequence->get_clip());

        auto concat_0 = std::make_shared<opset5::Concat>(OutputVector{rnn_sequence_forward->output(0),
                                                                      rnn_sequence_reverse->output(0)}, 1);
        auto concat_1 = std::make_shared<opset5::Concat>(OutputVector{rnn_sequence_forward->output(1),
                                                                      rnn_sequence_reverse->output(1)}, 1);
        ngraph::copy_runtime_info(rnn_sequence, {H, W, R, B, rnn_sequence_forward, rnn_sequence_reverse,
                                                 concat_0, concat_1});
        concat_0->set_friendly_name(rnn_sequence->get_friendly_name() + ".0");
        concat_1->set_friendly_name(rnn_sequence->get_friendly_name() + ".1");
        ngraph::replace_node(rnn_sequence, {concat_0->output(0), concat_1->output(0)});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_sequence_ngraph, "BidirectionalRNNSequenceDecomposition");
    this->register_matcher(m, callback);
}
