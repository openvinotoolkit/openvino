// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sequence_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov::element;
using namespace ov::op::util;

namespace {
bool is_equal_consts(const shared_ptr<ov::Node>& l, const shared_ptr<ov::Node>& r) {
    auto l_const = ov::as_type_ptr<ov::op::v0::Constant>(l);
    auto r_const = ov::as_type_ptr<ov::op::v0::Constant>(r);
    if (l_const && r_const) {
        auto l_ptr = l_const->get_data_ptr();
        auto r_ptr = r_const->get_data_ptr();
        size_t bytes = shape_size(l_const->get_shape()) * l_const->get_element_type().size();
        return l_const->get_element_type() == r_const->get_element_type() &&
               l_const->get_shape() == r_const->get_shape() && (l_ptr == r_ptr || memcmp(l_ptr, r_ptr, bytes) == 0);
    }
    return false;
}

bool check_WRB(const shared_ptr<RNNCellBase>& cell_1, const shared_ptr<RNNCellBase>& cell_2) {
    int64_t idx_W = 2, idx_R = 3, idx_B = 4;
    auto increase_indexes = [&]() {
        ++idx_B;
        ++idx_R;
        ++idx_W;
    };
    auto lstm_cell_v4_1 = ov::as_type_ptr<ov::op::v4::LSTMCell>(cell_1);
    auto lstm_cell_v4_2 = ov::as_type_ptr<ov::op::v4::LSTMCell>(cell_2);
    // 2nd input is Cell State
    if (lstm_cell_v4_1 && lstm_cell_v4_2) {
        increase_indexes();
    }
    auto lstm_cell_v0_1 = ov::as_type_ptr<ov::op::v0::LSTMCell>(cell_1);
    auto lstm_cell_v0_2 = ov::as_type_ptr<ov::op::v0::LSTMCell>(cell_2);
    if (lstm_cell_v0_1 && lstm_cell_v0_2) {
        if (lstm_cell_v0_1->get_weights_format() != lstm_cell_v0_2->get_weights_format() ||
            lstm_cell_v0_1->get_input_forget() != lstm_cell_v0_2->get_input_forget()) {
            return false;
        }
        increase_indexes();
    }
    auto lW = cell_1->input_value(idx_W).get_node_shared_ptr();
    auto lR = cell_1->input_value(idx_R).get_node_shared_ptr();
    auto lB = cell_1->input_value(idx_B).get_node_shared_ptr();
    auto rW = cell_2->input_value(idx_W).get_node_shared_ptr();
    auto rR = cell_2->input_value(idx_R).get_node_shared_ptr();
    auto rB = cell_2->input_value(idx_B).get_node_shared_ptr();
    bool is_equal = (lW.get() == rW.get() || is_equal_consts(lW, rW));
    is_equal = is_equal && (lR.get() == rR.get() || is_equal_consts(lR, rR));
    is_equal = is_equal && (lB.get() == rB.get() || is_equal_consts(lB, rB));
    return is_equal;
}

bool is_equal_cells(const shared_ptr<RNNCellBase>& cell_1, const shared_ptr<RNNCellBase>& cell_2) {
    bool is_equal = true;
    auto gru_cell_1 = ov::as_type_ptr<ov::op::v3::GRUCell>(cell_1);
    auto gru_cell_2 = ov::as_type_ptr<ov::op::v3::GRUCell>(cell_2);
    if (gru_cell_1 && gru_cell_2) {
        is_equal = gru_cell_1->get_linear_before_reset() == gru_cell_2->get_linear_before_reset();
    }
    is_equal = is_equal && cell_1->get_type_name() == cell_2->get_type_name() &&
               cell_1->get_hidden_size() == cell_2->get_hidden_size() &&
               cell_1->get_activations() == cell_2->get_activations() &&
               cell_1->get_activations_alpha() == cell_2->get_activations_alpha() &&
               cell_1->get_activations_beta() == cell_2->get_activations_beta() &&
               cell_1->get_clip() == cell_2->get_clip() && check_WRB(cell_1, cell_2);
    return is_equal;
}

bool check_lstm_cell(const shared_ptr<RNNCellBase>& prev_cell, const shared_ptr<RNNCellBase>& current_cell) {
    // check intermediate C outputs in case of LSTMCell
    // LSTMCell - C -> LSTMCell
    if ((ov::as_type_ptr<ov::op::v4::LSTMCell>(prev_cell) || ov::as_type_ptr<ov::op::v0::LSTMCell>(prev_cell))) {
        const auto& target_inputs = prev_cell->get_output_target_inputs(1);
        bool valid = target_inputs.empty() ||
                     (target_inputs.size() == 1 &&
                      ov::as_type<RNNCellBase>(target_inputs.begin()->get_node()) == current_cell.get() &&
                      target_inputs.begin()->get_index() == 2);

        // if intermediate C output is connected to other node, except ov::op::v4::LSTMCell,
        // we can't replace cells with sequence. Sequence doesn't provide access to these outputs.
        return valid;
    }
    return true;
}

shared_ptr<RNNCellBase> find_cell_chain(ov::pass::NodeRegistry& cp_from,
                                        ov::pass::NodeRegistry& cp_to,
                                        const shared_ptr<RNNCellBase>& current_cell,
                                        ov::OutputVector& x_to_concat,
                                        ov::OutputVector& attention_to_concat,
                                        map<int, ov::Output<ov::Node>>& h_outputs_to_redirect,
                                        int& cells_cnt,
                                        const shared_ptr<ov::Node>& axis_1) {
    cells_cnt = 1;
    shared_ptr<RNNCellBase> current = current_cell;
    while (true) {
        cp_from.add(current);
        // check the source node of HiddenState input
        auto prev = current->input_value(1).get_node_shared_ptr();
        auto prev_cell = ov::as_type_ptr<RNNCellBase>(prev);

        auto in_X = current->input(0);
        x_to_concat.push_back(cp_to.make<ov::op::v0::Unsqueeze>(in_X.get_source_output(), axis_1));
        h_outputs_to_redirect[cells_cnt] = current->output(0);

        if (auto augru = ov::as_type_ptr<ov::op::internal::AUGRUCell>(current)) {
            attention_to_concat.push_back(cp_to.make<ov::op::v0::Unsqueeze>(augru->input_value(5), axis_1));
        }

        if (prev_cell && is_equal_cells(prev_cell, current) && check_lstm_cell(prev_cell, current)) {
            current = prev_cell;
            cells_cnt++;
        } else {
            break;
        }
    }
    reverse(x_to_concat.begin(), x_to_concat.end());
    reverse(attention_to_concat.begin(), attention_to_concat.end());
    // the first cell in the chain
    return current;
}

bool create_sequence(ov::pass::NodeRegistry& cp_to,
                     const shared_ptr<RNNCellBase>& first_cell,
                     const shared_ptr<RNNCellBase>& last_cell,
                     const ov::OutputVector& x_to_concat,
                     const ov::OutputVector& attention_to_concat,
                     const map<int, ov::Output<ov::Node>>& h_outputs_to_redirect,
                     int cells_cnt,
                     const shared_ptr<ov::Node>& axis_0,
                     const shared_ptr<ov::Node>& axis_1) {
    int64_t idx_W = 2, idx_R = 3, idx_B = 4;
    // 2nd input is Cell State
    bool is_lstm = false;
    if (ov::as_type_ptr<ov::op::v4::LSTMCell>(last_cell) || ov::as_type_ptr<ov::op::v0::LSTMCell>(last_cell)) {
        is_lstm = true;
        idx_B++;
        idx_R++;
        idx_W++;
    }

    const auto X_in = cp_to.make<ov::op::v0::Concat>(x_to_concat, 1);
    const auto Ht_in = cp_to.make<ov::op::v0::Unsqueeze>(first_cell->input_value(1), axis_1);
    const auto W_in = cp_to.make<ov::op::v0::Unsqueeze>(first_cell->input_value(idx_W), axis_0);
    const auto R_in = cp_to.make<ov::op::v0::Unsqueeze>(first_cell->input_value(idx_R), axis_0);
    const auto B_in = cp_to.make<ov::op::v0::Unsqueeze>(first_cell->input_value(idx_B), axis_0);

    const auto& shape_node = cp_to.add(ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(first_cell->input_value(0)));
    const auto& zero = cp_to.make<ov::op::v0::Constant>(i64, ov::Shape{1}, 0);
    const auto& batch_dimension = cp_to.add(ov::op::util::make_try_fold<ov::op::v8::Gather>(shape_node, zero, axis_0));
    auto seq_lengths_scalar = cp_to.make<ov::op::v0::Constant>(i64, ov::Shape{}, cells_cnt);
    auto sequence_lengths_in =
        cp_to.add(ov::op::util::make_try_fold<ov::op::v3::Broadcast>(seq_lengths_scalar, batch_dimension));
    shared_ptr<ov::Node> sequence;
    ov::OutputVector outputs(1);
    if (ov::as_type_ptr<ov::op::v4::LSTMCell>(first_cell)) {
        const auto Ct_in = cp_to.make<ov::op::v0::Unsqueeze>(first_cell->input_value(2), axis_1);
        sequence = cp_to.make<ov::op::v5::LSTMSequence>(X_in,
                                                        Ht_in,
                                                        Ct_in,
                                                        sequence_lengths_in,
                                                        W_in,
                                                        R_in,
                                                        B_in,
                                                        first_cell->get_hidden_size(),
                                                        ov::op::RecurrentSequenceDirection::FORWARD,
                                                        first_cell->get_activations_alpha(),
                                                        first_cell->get_activations_beta(),
                                                        first_cell->get_activations(),
                                                        first_cell->get_clip());
        outputs.resize(2);
        outputs[1] = cp_to.make<ov::op::v0::Squeeze>(sequence->output(2), axis_1);
    } else if (auto lstm_cell_v0 = ov::as_type_ptr<ov::op::v0::LSTMCell>(first_cell)) {
        // input_forget modification is not supported
        if (lstm_cell_v0->get_input_forget()) {
            return false;
        }
        auto weights_format = lstm_cell_v0->get_weights_format();
        ov::Output<ov::Node> W = W_in, R = R_in, B = B_in;
        if (weights_format != ov::op::LSTMWeightsFormat::FICO) {
            W = ov::op::util::convert_lstm_node_format(W_in, convert_lstm_weights_enums(weights_format));
            R = ov::op::util::convert_lstm_node_format(R_in, convert_lstm_weights_enums(weights_format));
            B = ov::op::util::convert_lstm_node_format(B_in, convert_lstm_weights_enums(weights_format));
        }
        const auto Ct_in = cp_to.make<ov::op::v0::Unsqueeze>(first_cell->input_value(2), axis_1);
        sequence = cp_to.make<ov::op::v5::LSTMSequence>(X_in,
                                                        Ht_in,
                                                        Ct_in,
                                                        sequence_lengths_in,
                                                        W,
                                                        R,
                                                        B,
                                                        first_cell->get_hidden_size(),
                                                        ov::op::RecurrentSequenceDirection::FORWARD,
                                                        first_cell->get_activations_alpha(),
                                                        first_cell->get_activations_beta(),
                                                        first_cell->get_activations(),
                                                        first_cell->get_clip());
        outputs.resize(2);
        outputs[1] = cp_to.make<ov::op::v0::Squeeze>(sequence->output(2), axis_1);
    } else if (auto gru_cell = ov::as_type_ptr<ov::op::v3::GRUCell>(first_cell)) {
        sequence = cp_to.make<ov::op::v5::GRUSequence>(X_in,
                                                       Ht_in,
                                                       sequence_lengths_in,
                                                       W_in,
                                                       R_in,
                                                       B_in,
                                                       first_cell->get_hidden_size(),
                                                       ov::op::RecurrentSequenceDirection::FORWARD,
                                                       first_cell->get_activations(),
                                                       first_cell->get_activations_alpha(),
                                                       first_cell->get_activations_beta(),
                                                       first_cell->get_clip(),
                                                       gru_cell->get_linear_before_reset());
    } else if (ov::as_type_ptr<ov::op::v0::RNNCell>(first_cell)) {
        sequence = cp_to.make<ov::op::v5::RNNSequence>(X_in,
                                                       Ht_in,
                                                       sequence_lengths_in,
                                                       W_in,
                                                       R_in,
                                                       B_in,
                                                       first_cell->get_hidden_size(),
                                                       ov::op::RecurrentSequenceDirection::FORWARD,
                                                       first_cell->get_activations(),
                                                       first_cell->get_activations_alpha(),
                                                       first_cell->get_activations_beta(),
                                                       first_cell->get_clip());
    } else if (ov::as_type_ptr<ov::op::internal::AUGRUCell>(first_cell)) {
        const auto A_in = cp_to.make<ov::op::v0::Concat>(attention_to_concat, 1);
        sequence = cp_to.make<ov::op::internal::AUGRUSequence>(X_in,
                                                               Ht_in,
                                                               sequence_lengths_in,
                                                               W_in,
                                                               R_in,
                                                               B_in,
                                                               A_in,
                                                               first_cell->get_hidden_size());
    } else {
        // cell is not supported;
        return false;
    }

    if (!h_outputs_to_redirect.empty()) {
        auto squeeze_Y = cp_to.make<ov::op::v0::Squeeze>(sequence->output(0), axis_1);
        auto split = cp_to.make<ov::op::v1::Split>(squeeze_Y, axis_1, cells_cnt);

        for (auto it : h_outputs_to_redirect) {
            auto Hi = split->output(cells_cnt - it.first);
            auto friendly_name = it.second.get_node_shared_ptr()->get_friendly_name();
            if (it.first == 1) {
                Hi = sequence->output(1);
            }
            auto squeeze = cp_to.make<ov::op::v0::Squeeze>(Hi, axis_1);
            it.second.replace(squeeze);
            if (is_lstm) {
                friendly_name += ":1";
            }
            squeeze->set_friendly_name(friendly_name);
        }
    }
    if (is_lstm) {
        auto squeeze = cp_to.make<ov::op::v0::Squeeze>(sequence->output(2), axis_1);
        last_cell->output(1).replace(squeeze);
        squeeze->set_friendly_name(last_cell->get_friendly_name() + ":2");
    }

    return true;
}
}  // namespace

ov::pass::SequenceFusion::SequenceFusion() {
    MATCHER_SCOPE(SequenceFusion);

    auto cell = pattern::wrap_type<RNNCellBase>();
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry copy_from;
        NodeRegistry copy_to;
        auto cell = m.get_match_root();
        shared_ptr<RNNCellBase> current_cell = ov::as_type_ptr<RNNCellBase>(cell);
        if (!current_cell) {
            return false;
        }
        // check that this is the last Cell in the chain, e.g.
        // GRUCell -> GRUCell (the last cell) -> OtherNode
        // GRUCell (hidden_size = 128) -> GRUCell (hs = 128, the last) -> GRUCell (hs = 64)
        for (const auto& target : cell->get_output_target_inputs(0)) {
            auto cell_1 = ov::as_type_ptr<RNNCellBase>(target.get_node()->shared_from_this());
            if (cell_1 && is_equal_cells(cell_1, current_cell)) {
                return false;
            }
        }
        int cells_cnt;
        ov::OutputVector x_to_concat;
        ov::OutputVector attention_to_concat;
        map<int, ov::Output<ov::Node>> h_outputs_to_redirect;
        auto axis_0 = copy_to.make<ov::op::v0::Constant>(i64, Shape{}, 0);
        auto axis_1 = copy_to.make<ov::op::v0::Constant>(i64, Shape{}, 1);

        // detect chain (Cell->Cell->Cell->..)
        auto first_cell = find_cell_chain(copy_from,
                                          copy_to,
                                          current_cell,
                                          x_to_concat,
                                          attention_to_concat,
                                          h_outputs_to_redirect,
                                          cells_cnt,
                                          axis_1);
        if (!first_cell) {
            return false;
        }

        // no reasons to create sequence if the single cell detected.
        // TODO: investigate optimal cnt of cells
        constexpr int optimal_cnt_of_cells = 2;
        if (cells_cnt < optimal_cnt_of_cells) {
            return false;
        }

        auto res = create_sequence(copy_to,
                                   first_cell,
                                   current_cell,
                                   x_to_concat,
                                   attention_to_concat,
                                   h_outputs_to_redirect,
                                   cells_cnt,
                                   axis_0,
                                   axis_1);
        if (!res) {
            return false;
        }
        copy_runtime_info(copy_from.get(), copy_to.get());
        return true;
    };

    auto m = make_shared<pattern::Matcher>(cell, matcher_name);
    this->register_matcher(m, callback);
}
