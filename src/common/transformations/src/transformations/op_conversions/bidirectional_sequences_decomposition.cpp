// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::BidirectionalLSTMSequenceDecomposition::BidirectionalLSTMSequenceDecomposition() {
    MATCHER_SCOPE(BidirectionalLSTMSequenceDecomposition);
    auto lstm_sequence_ov = ov::pass::pattern::wrap_type<ov::op::v5::LSTMSequence>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto lstm_sequence = ov::as_type_ptr<ov::op::v5::LSTMSequence>(m.get_match_root());
        if (!lstm_sequence || transformation_callback(lstm_sequence)) {
            return false;
        }

        if (lstm_sequence->get_direction() != ov::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        auto axis_0 = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
        auto H = std::make_shared<ov::op::v1::Split>(lstm_sequence->input_value(1), axis_1, 2);
        auto C = std::make_shared<ov::op::v1::Split>(lstm_sequence->input_value(2), axis_1, 2);
        auto W = std::make_shared<ov::op::v1::Split>(lstm_sequence->input_value(4), axis_0, 2);
        auto R = std::make_shared<ov::op::v1::Split>(lstm_sequence->input_value(5), axis_0, 2);
        auto B = std::make_shared<ov::op::v1::Split>(lstm_sequence->input_value(6), axis_0, 2);
        auto lstm_sequence_forward =
            std::make_shared<ov::op::v5::LSTMSequence>(lstm_sequence->input_value(0),
                                                       H->output(0),
                                                       C->output(0),
                                                       lstm_sequence->input_value(3),
                                                       W->output(0),
                                                       R->output(0),
                                                       B->output(0),
                                                       lstm_sequence->get_hidden_size(),
                                                       ov::op::RecurrentSequenceDirection::FORWARD,
                                                       lstm_sequence->get_activations_alpha(),
                                                       lstm_sequence->get_activations_beta(),
                                                       lstm_sequence->get_activations(),
                                                       lstm_sequence->get_clip());

        auto lstm_sequence_reverse =
            std::make_shared<ov::op::v5::LSTMSequence>(lstm_sequence->input_value(0),
                                                       H->output(1),
                                                       C->output(1),
                                                       lstm_sequence->input_value(3),
                                                       W->output(1),
                                                       R->output(1),
                                                       B->output(1),
                                                       lstm_sequence->get_hidden_size(),
                                                       ov::op::RecurrentSequenceDirection::REVERSE,
                                                       lstm_sequence->get_activations_alpha(),
                                                       lstm_sequence->get_activations_beta(),
                                                       lstm_sequence->get_activations(),
                                                       lstm_sequence->get_clip());

        auto concat_0 = std::make_shared<ov::op::v0::Concat>(
            OutputVector{lstm_sequence_forward->output(0), lstm_sequence_reverse->output(0)},
            1);
        auto concat_1 = std::make_shared<ov::op::v0::Concat>(
            OutputVector{lstm_sequence_forward->output(1), lstm_sequence_reverse->output(1)},
            1);
        auto concat_2 = std::make_shared<ov::op::v0::Concat>(
            OutputVector{lstm_sequence_forward->output(2), lstm_sequence_reverse->output(2)},
            1);
        ov::copy_runtime_info(
            lstm_sequence,
            {H, C, W, R, B, lstm_sequence_forward, lstm_sequence_reverse, concat_0, concat_1, concat_2});
        concat_0->set_friendly_name(lstm_sequence->get_friendly_name() + ".0");
        concat_1->set_friendly_name(lstm_sequence->get_friendly_name() + ".1");
        concat_2->set_friendly_name(lstm_sequence->get_friendly_name() + ".2");
        ov::replace_node(lstm_sequence, {concat_0->output(0), concat_1->output(0), concat_2->output(0)});
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(lstm_sequence_ov, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::BidirectionalGRUSequenceDecomposition::BidirectionalGRUSequenceDecomposition() {
    MATCHER_SCOPE(BidirectionalGRUSequenceDecomposition);
    auto gru_sequence_ov = ov::pass::pattern::wrap_type<ov::op::v5::GRUSequence>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto gru_sequence = ov::as_type_ptr<ov::op::v5::GRUSequence>(m.get_match_root());
        if (!gru_sequence || transformation_callback(gru_sequence)) {
            return false;
        }

        if (gru_sequence->get_direction() != ov::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        auto axis_0 = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
        auto H = std::make_shared<ov::op::v1::Split>(gru_sequence->input_value(1), axis_1, 2);
        auto W = std::make_shared<ov::op::v1::Split>(gru_sequence->input_value(3), axis_0, 2);
        auto R = std::make_shared<ov::op::v1::Split>(gru_sequence->input_value(4), axis_0, 2);
        auto B = std::make_shared<ov::op::v1::Split>(gru_sequence->input_value(5), axis_0, 2);
        auto gru_sequence_forward =
            std::make_shared<ov::op::v5::GRUSequence>(gru_sequence->input_value(0),
                                                      H->output(0),
                                                      gru_sequence->input_value(2),
                                                      W->output(0),
                                                      R->output(0),
                                                      B->output(0),
                                                      gru_sequence->get_hidden_size(),
                                                      ov::op::RecurrentSequenceDirection::FORWARD,
                                                      gru_sequence->get_activations(),
                                                      gru_sequence->get_activations_alpha(),
                                                      gru_sequence->get_activations_beta(),
                                                      gru_sequence->get_clip(),
                                                      gru_sequence->get_linear_before_reset());

        auto gru_sequence_reverse =
            std::make_shared<ov::op::v5::GRUSequence>(gru_sequence->input_value(0),
                                                      H->output(1),
                                                      gru_sequence->input_value(2),
                                                      W->output(1),
                                                      R->output(1),
                                                      B->output(1),
                                                      gru_sequence->get_hidden_size(),
                                                      ov::op::RecurrentSequenceDirection::REVERSE,
                                                      gru_sequence->get_activations(),
                                                      gru_sequence->get_activations_alpha(),
                                                      gru_sequence->get_activations_beta(),
                                                      gru_sequence->get_clip(),
                                                      gru_sequence->get_linear_before_reset());

        auto concat_0 = std::make_shared<ov::op::v0::Concat>(
            OutputVector{gru_sequence_forward->output(0), gru_sequence_reverse->output(0)},
            1);
        auto concat_1 = std::make_shared<ov::op::v0::Concat>(
            OutputVector{gru_sequence_forward->output(1), gru_sequence_reverse->output(1)},
            1);
        ov::copy_runtime_info(gru_sequence,
                              {H, W, R, B, gru_sequence_forward, gru_sequence_reverse, concat_0, concat_1});
        concat_0->set_friendly_name(gru_sequence->get_friendly_name() + ".0");
        concat_1->set_friendly_name(gru_sequence->get_friendly_name() + ".1");
        ov::replace_node(gru_sequence, {concat_0->output(0), concat_1->output(0)});
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gru_sequence_ov, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::BidirectionalRNNSequenceDecomposition::BidirectionalRNNSequenceDecomposition() {
    MATCHER_SCOPE(BidirectionalRNNSequenceDecomposition);
    auto rnn_sequence_ov = ov::pass::pattern::wrap_type<ov::op::v5::RNNSequence>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto rnn_sequence = ov::as_type_ptr<ov::op::v5::RNNSequence>(m.get_match_root());
        if (!rnn_sequence || transformation_callback(rnn_sequence)) {
            return false;
        }

        if (rnn_sequence->get_direction() != ov::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        auto axis_0 = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
        auto H = std::make_shared<ov::op::v1::Split>(rnn_sequence->input_value(1), axis_1, 2);
        auto W = std::make_shared<ov::op::v1::Split>(rnn_sequence->input_value(3), axis_0, 2);
        auto R = std::make_shared<ov::op::v1::Split>(rnn_sequence->input_value(4), axis_0, 2);
        auto B = std::make_shared<ov::op::v1::Split>(rnn_sequence->input_value(5), axis_0, 2);
        auto rnn_sequence_forward =
            std::make_shared<ov::op::v5::RNNSequence>(rnn_sequence->input_value(0),
                                                      H->output(0),
                                                      rnn_sequence->input_value(2),
                                                      W->output(0),
                                                      R->output(0),
                                                      B->output(0),
                                                      rnn_sequence->get_hidden_size(),
                                                      ov::op::RecurrentSequenceDirection::FORWARD,
                                                      rnn_sequence->get_activations(),
                                                      rnn_sequence->get_activations_alpha(),
                                                      rnn_sequence->get_activations_beta(),
                                                      rnn_sequence->get_clip());

        auto rnn_sequence_reverse =
            std::make_shared<ov::op::v5::RNNSequence>(rnn_sequence->input_value(0),
                                                      H->output(1),
                                                      rnn_sequence->input_value(2),
                                                      W->output(1),
                                                      R->output(1),
                                                      B->output(1),
                                                      rnn_sequence->get_hidden_size(),
                                                      ov::op::RecurrentSequenceDirection::REVERSE,
                                                      rnn_sequence->get_activations(),
                                                      rnn_sequence->get_activations_alpha(),
                                                      rnn_sequence->get_activations_beta(),
                                                      rnn_sequence->get_clip());

        auto concat_0 = std::make_shared<ov::op::v0::Concat>(
            OutputVector{rnn_sequence_forward->output(0), rnn_sequence_reverse->output(0)},
            1);
        auto concat_1 = std::make_shared<ov::op::v0::Concat>(
            OutputVector{rnn_sequence_forward->output(1), rnn_sequence_reverse->output(1)},
            1);
        ov::copy_runtime_info(rnn_sequence,
                              {H, W, R, B, rnn_sequence_forward, rnn_sequence_reverse, concat_0, concat_1});
        concat_0->set_friendly_name(rnn_sequence->get_friendly_name() + ".0");
        concat_1->set_friendly_name(rnn_sequence->get_friendly_name() + ".1");
        ov::replace_node(rnn_sequence, {concat_0->output(0), concat_1->output(0)});
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rnn_sequence_ov, matcher_name);
    this->register_matcher(m, callback);
}
