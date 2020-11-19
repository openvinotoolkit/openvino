// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_sequences_to_sequences_ie.hpp"

#include <memory>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <legacy/ngraph_ops/lstm_sequence_ie.hpp>
#include <legacy/ngraph_ops/gru_sequence_ie.hpp>
#include <legacy/ngraph_ops/rnn_sequence_ie.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertLSTMSequenceMatcher, "ConvertLSTMSequenceMatcher", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertGRUSequenceMatcher, "ConvertGRUSequenceMatcher", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertRNNSequenceMatcher, "ConvertRNNSequenceMatcher", 0);

namespace {
    int64_t get_seq_axis(const std::shared_ptr<ngraph::Node>& sequence_node) {
        // Optimization.
        // Plug-ins support seq_axis attribute (value 1 or 0) for Seq ops, but according to the spec we don't
        // support this attribute and should insert Transpose layer before and after Seq op in TI to Sequences
        // transformation. Additional Transpose layers affect the performance, so we try to detect pattern
        // Transpose(axis_order={1,0,2}) -> Seq -> Transpose(axis_order={2,1,0,3}
        // and replace unnecessary Transpose ops with SeqIE (seq_axis = 0) to transfer value
        // of the attribute to plug-ins.
        // todo: specify seq_axis attribute for Sequence ops.
        int64_t seq_axis = 1; // default
        const auto& target_inputs = sequence_node->output(0).get_target_inputs();
        if (target_inputs.size() == 1) {
            const auto& transpose_before = std::dynamic_pointer_cast<ngraph::opset5::Transpose>(sequence_node->input_value(0).get_node_shared_ptr());
            const auto& transpose_after = std::dynamic_pointer_cast<ngraph::opset5::Transpose>(target_inputs.begin()->get_node()->shared_from_this());
            if (transpose_after != nullptr && transpose_before != nullptr) {
                auto order_before = std::dynamic_pointer_cast<ngraph::opset5::Constant>(
                        transpose_before->input_value(1).get_node_shared_ptr());
                auto order_after = std::dynamic_pointer_cast<ngraph::opset5::Constant>(
                        transpose_after->input_value(1).get_node_shared_ptr());
                if (order_before != nullptr && order_after != nullptr) {
                    auto order_before_values = order_before->cast_vector<int64_t>();
                    auto order_after_values = order_after->cast_vector<int64_t>();
                    std::vector<int64_t> order_ref_before = {1, 0, 2};
                    std::vector<int64_t> order_ref_after = {2, 1, 0, 3};
                    if (order_before_values == order_ref_before && order_after_values == order_ref_after) {
                        seq_axis = 0;
                    }
                }
            }
        }
        return seq_axis;
    }
} // namespace

ngraph::pass::ConvertLSTMSequenceMatcher::ConvertLSTMSequenceMatcher() {
    auto lstm_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::opset5::LSTMSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto lstm_sequence = std::dynamic_pointer_cast<ngraph::opset5::LSTMSequence>(m.get_match_root());
        if (!lstm_sequence) {
            return false;
        }

        const auto& W = lstm_sequence->input_value(4);
        const auto& R = lstm_sequence->input_value(5);

        // Bidirectional cases are not supported
        if (lstm_sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        // Detect pattern: Transpose_before -> Seq -> Transpose_after
        auto seq_axis = get_seq_axis(lstm_sequence);
        ngraph::Output<ngraph::Node> in_0 = lstm_sequence->input(0).get_source_output();
        if (seq_axis == 0) {
            // input(0) to Transpose_before
            in_0 = lstm_sequence->get_input_source_output(0).get_node_shared_ptr()->get_input_source_output(0);
        }
        // for forward/reverse cases we can squeeze num_direction dimension
        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset5::Squeeze>(lstm_sequence->input_value(1), axis_1);
        auto in_2 = std::make_shared<ngraph::opset5::Squeeze>(lstm_sequence->input_value(2), axis_1);
        auto concat = std::make_shared<ngraph::opset5::Concat>(ngraph::OutputVector{W, R}, 2);
        auto axis_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset5::Squeeze>(concat->output(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset5::Squeeze>(lstm_sequence->input_value(6), axis_2);
        auto lstm_sequence_ie = std::make_shared<ngraph::op::LSTMSequenceIE>(
                in_0,  // X
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
                lstm_sequence->get_clip(),
                seq_axis);

        auto unsqueeze_axis = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_1 = std::make_shared<ngraph::opset5::Unsqueeze>(lstm_sequence_ie->output(0), unsqueeze_axis);
        auto unsqueeze_2 = std::make_shared<ngraph::opset5::Unsqueeze>(lstm_sequence_ie->output(1), unsqueeze_axis);
        auto unsqueeze_3 = std::make_shared<ngraph::opset5::Unsqueeze>(lstm_sequence_ie->output(2), unsqueeze_axis);

        ngraph::copy_runtime_info(lstm_sequence, {concat, lstm_sequence_ie, in_1, in_2, in_3, in_4, unsqueeze_1,
                                                  unsqueeze_2, unsqueeze_3});
        unsqueeze_1->set_friendly_name(lstm_sequence->get_friendly_name()+".0");
        unsqueeze_2->set_friendly_name(lstm_sequence->get_friendly_name()+".1");
        unsqueeze_3->set_friendly_name(lstm_sequence->get_friendly_name()+".2");
        if (seq_axis == 1) {
            ngraph::replace_node(lstm_sequence, {unsqueeze_1->output(0), unsqueeze_2->output(0), unsqueeze_3->output(0)});
        } else {
            auto transpose_after = lstm_sequence->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
            ngraph::replace_node(transpose_after, unsqueeze_1);
            ngraph::replace_node(lstm_sequence, {lstm_sequence_ie->output(0), unsqueeze_2->output(0), unsqueeze_3->output(0)});
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstm_sequence_ngraph, "ConvertLSTMSequenceToLSTMSequenceIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertGRUSequenceMatcher::ConvertGRUSequenceMatcher() {
    auto gru_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::opset5::GRUSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto gru_sequence = std::dynamic_pointer_cast<ngraph::opset5::GRUSequence>(m.get_match_root());
        if (!gru_sequence) {
            return false;
        }

        auto W = gru_sequence->input_value(3);
        auto R = gru_sequence->input_value(4);

        // Bidirectional cases are not supported
        if (gru_sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        // Detect pattern: Transpose_before -> Seq -> Transpose_after
        auto seq_axis = get_seq_axis(gru_sequence);
        ngraph::Output<ngraph::Node> in_0 = gru_sequence->input(0).get_source_output();
        if (seq_axis == 0) {
            // input(0) to Transpose_before
            in_0 = gru_sequence->get_input_source_output(0).get_node_shared_ptr()->get_input_source_output(0);
        }
        // for forward/reverse cases we can squeeze num_direction dimension
        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset5::Squeeze>(gru_sequence->input_value(1), axis_1);
        auto concat = std::make_shared<ngraph::opset5::Concat>(ngraph::OutputVector{W, R}, 2);
        auto axis_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset5::Squeeze>(concat->output(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset5::Squeeze>(gru_sequence->input_value(5), axis_2);

        auto gru_sequence_ie = std::make_shared<ngraph::op::GRUSequenceIE>(
                in_0, // X
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
                gru_sequence->get_linear_before_reset(),
                seq_axis);

        auto unsqueeze_axis = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_1 = std::make_shared<ngraph::opset5::Unsqueeze>(gru_sequence_ie->output(0), unsqueeze_axis);
        auto unsqueeze_2 = std::make_shared<ngraph::opset5::Unsqueeze>(gru_sequence_ie->output(1), unsqueeze_axis);

        ngraph::copy_runtime_info(gru_sequence, {concat, gru_sequence_ie, unsqueeze_1, unsqueeze_2, in_1, in_3, in_4});
        unsqueeze_1->set_friendly_name(gru_sequence->get_friendly_name()+".0");
        unsqueeze_2->set_friendly_name(gru_sequence->get_friendly_name()+".1");
        if (seq_axis == 1) {
            ngraph::replace_node(gru_sequence, {unsqueeze_1->output(0), unsqueeze_2->output(0)});
        } else {
            auto transpose_after = gru_sequence->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
            ngraph::replace_node(transpose_after, unsqueeze_1);
            ngraph::replace_node(gru_sequence, {gru_sequence_ie->output(0), unsqueeze_2->output(0)});
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gru_sequence_ngraph, "ConvertGRUSequenceToGRUSequenceIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertRNNSequenceMatcher::ConvertRNNSequenceMatcher() {
    auto rnn_sequence_ngraph = ngraph::pattern::wrap_type<ngraph::opset5::RNNSequence>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto rnn_sequence = std::dynamic_pointer_cast<ngraph::opset5::RNNSequence>(m.get_match_root());
        if (!rnn_sequence) {
            return false;
        }

        // Bidirectional cases are not supported
        if (rnn_sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        // Detect pattern: Transpose_before -> Seq -> Transpose_after
        auto seq_axis = get_seq_axis(rnn_sequence);
        ngraph::Output<ngraph::Node> in_0 = rnn_sequence->input(0).get_source_output();
        if (seq_axis == 0) {
            // input(0) to Transpose_before
            in_0 = rnn_sequence->get_input_source_output(0).get_node_shared_ptr()->get_input_source_output(0);
        }

        auto W = rnn_sequence->input_value(3);
        auto R = rnn_sequence->input_value(4);

        // for forward/reverse cases we can squeeze num_direction dimension
        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset5::Squeeze>(rnn_sequence->input_value(1), axis_1);
        auto concat = std::make_shared<ngraph::opset5::Concat>(ngraph::OutputVector{W, R}, 2);
        auto axis_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset5::Squeeze>(concat->output(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset5::Squeeze>(rnn_sequence->input_value(5), axis_2);
        auto rnn_sequence_ie = std::make_shared<ngraph::op::RNNSequenceIE>(
                in_0,  // X
                in_1,  // initial_hidden_state
                rnn_sequence->input_value(2),
                in_3,  // WR
                in_4,  // B
                rnn_sequence->get_hidden_size(),
                rnn_sequence->get_direction(),
                rnn_sequence->get_activations(),
                rnn_sequence->get_activations_alpha(),
                rnn_sequence->get_activations_beta(),
                rnn_sequence->get_clip(),
                seq_axis);

        auto unsqueeze_axis = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_1 = std::make_shared<ngraph::opset5::Unsqueeze>(rnn_sequence_ie->output(0), unsqueeze_axis);
        auto unsqueeze_2 = std::make_shared<ngraph::opset5::Unsqueeze>(rnn_sequence_ie->output(1), unsqueeze_axis);

        ngraph::copy_runtime_info(rnn_sequence, {concat, rnn_sequence_ie, in_1, in_3, in_4, unsqueeze_1,
                                                 unsqueeze_2});
        unsqueeze_1->set_friendly_name(rnn_sequence->get_friendly_name()+".0");
        unsqueeze_2->set_friendly_name(rnn_sequence->get_friendly_name()+".1");

        if (seq_axis == 1) {
            ngraph::replace_node(rnn_sequence, {unsqueeze_1->output(0), unsqueeze_2->output(0)});
        } else {
            auto transpose_after = rnn_sequence->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
            ngraph::replace_node(transpose_after, unsqueeze_1);
            ngraph::replace_node(rnn_sequence, {rnn_sequence_ie->output(0), unsqueeze_2->output(0)});
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_sequence_ngraph, "ConvertRNNSequenceToRNNSequenceIE");
    this->register_matcher(m, callback);
}