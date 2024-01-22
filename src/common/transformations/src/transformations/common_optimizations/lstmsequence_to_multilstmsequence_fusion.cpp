// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/lstmsequence_to_multilstmsequence_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multi_lstm_sequence.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/rnn_cell_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

bool is_equal_cells(const std::shared_ptr<ov::op::v5::LSTMSequence>& lstm_1,
                    const std::shared_ptr<ov::op::v5::LSTMSequence>& lstm_2) {
    bool is_equal =
        lstm_1->get_type_name() == lstm_2->get_type_name() && lstm_1->get_hidden_size() == lstm_2->get_hidden_size() &&
        lstm_1->get_activations() == lstm_2->get_activations() &&
        lstm_1->get_activations_alpha() == lstm_2->get_activations_alpha() &&
        lstm_1->get_activations_beta() == lstm_2->get_activations_beta() && lstm_1->get_clip() == lstm_2->get_clip();
    return is_equal;
}

std::shared_ptr<ov::op::v5::LSTMSequence> find_lstm_chain(ov::pass::NodeRegistry& cp_from,
                                                          ov::pass::NodeRegistry& cp_to,
                                                          const std::shared_ptr<ov::op::v5::LSTMSequence>& current_lstm,
                                                          ov::OutputVector& x_to_concat,
                                                          ov::OutputVector& weights_to_concat,
                                                          ov::OutputVector& recurrence_weights_to_concat,
                                                          std::map<int, ov::Output<ov::Node>>& h_outputs_to_redirect,
                                                          int& lstm_count,
                                                          const std::shared_ptr<ov::Node>& axis_1) {
    lstm_count = 1;
    std::shared_ptr<ov::op::v5::LSTMSequence> current = current_lstm;
    while (true) {
        cp_from.add(current);

        // go to the Squeeze node
        auto prev_squeeze = current->input_value(0).get_node_shared_ptr();
        cp_from.add(prev_squeeze);
        auto prev_squeeze_ptr = std::dynamic_pointer_cast<ov::op::v0::Squeeze>(prev_squeeze);

        if (!prev_squeeze_ptr) {
            break;
        }

        // go to actual LSTM
        auto prev_lstm = prev_squeeze_ptr->input_value(0).get_node_shared_ptr();
        auto prev_lstm_ptr = std::dynamic_pointer_cast<ov::op::v5::LSTMSequence>(prev_lstm);

        if (!prev_lstm_ptr) {
            break;
        }

        auto in_X = current->input(0);
        x_to_concat.push_back(cp_to.make<ov::op::v0::Unsqueeze>(in_X.get_source_output(), axis_1));
        //h_outputs_to_redirect[lstm_count] = prev_lstm->output(0);

        // TODO: check which inputs/attributes to redirect
        if (auto rnncell = std::dynamic_pointer_cast<ov::op::util::RNNCellBase>(current)) {
            weights_to_concat.push_back(cp_to.make<ov::op::v0::Unsqueeze>(rnncell->input_value(5), axis_1));
            recurrence_weights_to_concat.push_back(cp_to.make<ov::op::v0::Unsqueeze>(rnncell->input_value(6), axis_1));
        }

        if (prev_lstm_ptr && is_equal_cells(prev_lstm_ptr, current)) {
            current = prev_lstm_ptr;
            lstm_count++;
        } else {
            break;
        }
    }
    reverse(x_to_concat.begin(), x_to_concat.end());
    reverse(weights_to_concat.begin(), weights_to_concat.end());
    reverse(recurrence_weights_to_concat.begin(), recurrence_weights_to_concat.end());
    // the topographically first LSTM in the chain
    return current;
}

bool create_sequence(ov::pass::NodeRegistry& cp_to,
                     const std::shared_ptr<ov::op::v5::LSTMSequence>& first_cell,
                     const std::shared_ptr<ov::op::v5::LSTMSequence>& last_cell,
                     const ov::OutputVector& x_to_concat,
                     const ov::OutputVector& weights_to_concat,
                     const ov::OutputVector& recurrence_weights_to_concat,
                     const std::map<int, ov::Output<ov::Node>>& h_outputs_to_redirect,
                     int lstm_count,
                     const std::shared_ptr<ov::Node>& axis_0,
                     const std::shared_ptr<ov::Node>& axis_1) {
    const auto X_in = cp_to.make<ov::op::v0::Concat>(x_to_concat, 1);
    const auto X_in_squeezed = cp_to.make<ov::op::v0::Squeeze>(X_in->output(0), axis_1);
    const auto Ht_in = cp_to.add(first_cell->input_value(1).get_node_shared_ptr());
    const auto Ct_in = cp_to.add(first_cell->input_value(2).get_node_shared_ptr());
    const auto seq_len = cp_to.add(first_cell->input_value(3).get_node_shared_ptr());
    const auto W_in = cp_to.add(first_cell->input_value(4).get_node_shared_ptr());
    const auto R_in = cp_to.add(first_cell->input_value(5).get_node_shared_ptr());
    const auto B_in = cp_to.add(first_cell->input_value(6).get_node_shared_ptr());

    const auto& shape_node = cp_to.add(ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(first_cell->input_value(0)));
    const auto& zero = cp_to.make<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 0);
    const auto& batch_dimension = cp_to.add(ov::op::util::make_try_fold<ov::op::v8::Gather>(shape_node, zero, axis_0));
    std::shared_ptr<ov::Node> multi_lstm;
    ov::OutputVector outputs(1);
    if (std::dynamic_pointer_cast<ov::op::v5::LSTMSequence>(first_cell)) {
        multi_lstm = cp_to.make<ov::op::v13::MultiLSTMSequence>(X_in_squeezed,
                                                                Ht_in,
                                                                Ct_in,
                                                                seq_len,
                                                                W_in,
                                                                R_in,
                                                                B_in,
                                                                lstm_count,
                                                                first_cell->get_hidden_size(),
                                                                ov::op::RecurrentSequenceDirection::FORWARD,
                                                                ov::op::LSTMWeightsFormat::IFCO,
                                                                first_cell->get_activations_alpha(),
                                                                first_cell->get_activations_beta(),
                                                                first_cell->get_activations(),
                                                                first_cell->get_clip());
        outputs.resize(2);
        outputs[1] = cp_to.make<ov::op::v0::Squeeze>(multi_lstm->output(2), axis_1);
    }

    if (!h_outputs_to_redirect.empty()) {
        auto squeeze_Y = cp_to.make<ov::op::v0::Squeeze>(multi_lstm->output(0), axis_1);
        auto split = cp_to.make<ov::op::v1::Split>(squeeze_Y, axis_1, lstm_count);

        for (auto it : h_outputs_to_redirect) {
            auto Hi = split->output(lstm_count - it.first);
            auto friendly_name = it.second.get_node_shared_ptr()->get_friendly_name();
            if (it.first == 1) {
                Hi = multi_lstm->output(1);
            }
            auto squeeze = cp_to.make<ov::op::v0::Squeeze>(Hi, axis_1);
            it.second.replace(squeeze);
            squeeze->set_friendly_name(friendly_name);
        }
    }

    return true;
}

ov::pass::LSTMSequenceToMultiLSTMSequenceFusion::LSTMSequenceToMultiLSTMSequenceFusion() {
    MATCHER_SCOPE(LSTMSequenceToMultiLSTMSequenceFusion);

    auto lstm = pattern::wrap_type<op::v5::LSTMSequence>();
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry copy_from;
        NodeRegistry copy_to;
        auto lstm = m.get_match_root();
        std::shared_ptr<op::v5::LSTMSequence> current_lstm = std::dynamic_pointer_cast<op::v5::LSTMSequence>(lstm);
        if (!current_lstm) {
            return false;
        }
        // check if LSTMSequence's output doesn't lead to another LSTMSequence
        // essentially check if it's the last LSTMSequence in the series
        for (const auto& target : lstm->get_output_target_inputs(0)) {
            // detect Squeeze inbetween two LSTMSequence nodes
            auto squeeze = std::dynamic_pointer_cast<op::v0::Squeeze>(target.get_node()->shared_from_this());
            for (const auto& target_2 : squeeze->get_output_target_inputs(0)) {
                auto lstm_1 = std::dynamic_pointer_cast<op::v5::LSTMSequence>(target_2.get_node()->shared_from_this());
                if (squeeze && lstm_1 && is_equal_cells(lstm_1, current_lstm)) {
                    return false;
                }
            }
            auto lstm_node = squeeze->get_output_target_inputs(0);
        }

        int lstm_count;
        ov::OutputVector x_to_concat;
        ov::OutputVector weights_to_concat;
        ov::OutputVector recurrence_weights_to_concat;
        std::map<int, ov::Output<ov::Node>> h_outputs_to_redirect;
        auto axis_0 = copy_to.make<ov::op::v0::Constant>(ov::element::i64, Shape{}, 0);
        auto axis_1 = copy_to.make<ov::op::v0::Constant>(ov::element::i64, Shape{}, 1);

        // detect LSTMSequence chain (LSTM->Squeeze->LSTM->Squeeze->..)
        auto first_cell = find_lstm_chain(copy_from,
                                          copy_to,
                                          current_lstm,
                                          x_to_concat,
                                          weights_to_concat,
                                          recurrence_weights_to_concat,
                                          h_outputs_to_redirect,
                                          lstm_count,
                                          axis_1);
        if (!first_cell) {
            return false;
        }
        std::cout << "LSTM count: " << lstm_count << "\n";
        // no reasons to create Multi op if a single LSTM has been detected
        // TODO: determine optimal_cnt_of_lstms
        constexpr int optimal_cnt_of_lstms = 2;
        if (lstm_count < optimal_cnt_of_lstms) {
            return false;
        }
        auto res = create_sequence(copy_to,
                                   first_cell,
                                   current_lstm,
                                   x_to_concat,
                                   weights_to_concat,
                                   recurrence_weights_to_concat,
                                   h_outputs_to_redirect,
                                   lstm_count,
                                   axis_0,
                                   axis_1);
        if (!res) {
            return false;
        }
        copy_runtime_info(copy_from.get(), copy_to.get());
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(lstm, matcher_name);
    this->register_matcher(m, callback);
}
