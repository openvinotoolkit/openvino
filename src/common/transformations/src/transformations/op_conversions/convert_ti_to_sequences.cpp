// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_ti_to_sequences.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {
bool convertTensorIteratorToSequence(const std::shared_ptr<ov::op::v0::TensorIterator>& ti,
                                     const std::shared_ptr<ov::op::util::RNNCellBase>& found_cell,
                                     const ov::Output<ov::Node>& data,
                                     const ov::Output<ov::Node>& h_pattern,
                                     const ov::Output<ov::Node>& c_pattern,
                                     const ov::Output<ov::Node>& w_pattern,
                                     const ov::Output<ov::Node>& r_pattern,
                                     const ov::Output<ov::Node>& b_pattern,
                                     const ov::Output<ov::Node>& unsqueeze_after_cell) {
    const auto& func = ti->get_function();
    const auto& params = func->get_parameters();

    std::vector<std::shared_ptr<ov::op::v0::TensorIterator::InputDescription>> ordered_in_descs(3);
    int64_t stride = 0, slice_axis = 0;

    // Remember the order of the X and initial_hidden_state (+ initial_cell_state in case of LSTM) in the TensorIterator
    // params
    for (const auto& input_desc : ti->get_input_descriptions()) {
        auto param = params[input_desc->m_body_parameter_index];
        if (param == data.get_node_shared_ptr()) {
            auto slice_input = std::dynamic_pointer_cast<ov::op::v0::TensorIterator::SliceInputDescription>(input_desc);
            if (!slice_input)
                return false;

            stride = slice_input->m_stride;
            slice_axis = slice_input->m_axis;

            if (!(slice_axis == 0 || slice_axis == 1)) {
                return false;
            }
            ordered_in_descs[0] = input_desc;
        } else if (param == h_pattern.get_node_shared_ptr()) {
            ordered_in_descs[1] = input_desc;
        } else if (param == c_pattern.get_node_shared_ptr()) {
            ordered_in_descs[2] = input_desc;
        } else {
            return false;
        }
    }

    const auto& results = func->get_results();
    std::vector<std::shared_ptr<ov::op::v0::TensorIterator::OutputDescription>> ordered_out_descs(3);

    // Remember the order of cell outputs in the TensorIterator
    for (const auto& output_desc : ti->get_output_descriptions()) {
        std::shared_ptr<ov::op::v0::Result> res = results[output_desc->m_body_value_index];
        if (res->input_value(0) == unsqueeze_after_cell) {
            auto concat_output =
                std::dynamic_pointer_cast<ov::op::v0::TensorIterator::ConcatOutputDescription>(output_desc);
            if (!concat_output)
                return false;

            stride = concat_output->m_stride;
            ordered_out_descs[0] = output_desc;
        } else if (res->input_value(0) == found_cell->output(0)) {
            ordered_out_descs[1] = output_desc;
        } else if (found_cell->get_output_size() == 2 && res->input_value(0) == found_cell->output(1)) {
            ordered_out_descs[2] = output_desc;
        } else {
            return false;
        }
    }

    const auto ti_inputs = ti->input_values();
    auto X = ti_inputs[ordered_in_descs[0]->m_input_index];
    if (slice_axis == 0) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 0, 2});
        X = std::make_shared<ov::op::v1::Transpose>(ti_inputs[ordered_in_descs[0]->m_input_index], order);
    }

    // We must prepare cell inputs to sequence creation: insert num_directions elem via unsqueeze where needed (please,
    // see specification)
    auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto initial_hidden_state =
        std::make_shared<ov::op::v0::Unsqueeze>(ti_inputs[ordered_in_descs[1]->m_input_index], axis_1);

    // LSTM case
    std::shared_ptr<ov::Node> initial_cell_state =
        c_pattern.get_node_shared_ptr() == nullptr
            ? nullptr
            : std::make_shared<ov::op::v0::Unsqueeze>(ti_inputs[ordered_in_descs[2]->m_input_index], axis_1);

    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(X);
    auto batch_dimension =
        std::make_shared<ov::op::v1::Gather>(shape_of,
                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                             ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto seq_len_dim = std::make_shared<ov::op::v1::Gather>(shape_of,
                                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                            ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto seq_lengths = std::make_shared<ov::op::v3::Broadcast>(seq_len_dim, batch_dimension);
    auto axis_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto W = ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(w_pattern, axis_0);
    auto R = ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(r_pattern, axis_0);
    auto B = ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(b_pattern, axis_0);

    std::shared_ptr<ov::Node> sequence;
    if (ov::is_type<ov::op::v4::LSTMCell>(found_cell) || ov::is_type<ov::op::v0::LSTMCell>(found_cell)) {
        sequence = std::make_shared<ov::op::v5::LSTMSequence>(
            X,
            initial_hidden_state,
            initial_cell_state,
            seq_lengths,
            W,
            R,
            B,
            found_cell->get_hidden_size(),
            stride > 0 ? ov::op::RecurrentSequenceDirection::FORWARD : ov::op::RecurrentSequenceDirection::REVERSE,
            found_cell->get_activations_alpha(),
            found_cell->get_activations_beta(),
            found_cell->get_activations(),
            found_cell->get_clip());
    } else if (ov::is_type<ov::op::v0::RNNCell>(found_cell)) {
        sequence = std::make_shared<ov::op::v5::RNNSequence>(
            X,
            initial_hidden_state,
            seq_lengths,
            W,
            R,
            B,
            found_cell->get_hidden_size(),
            stride > 0 ? ov::op::RecurrentSequenceDirection::FORWARD : ov::op::RecurrentSequenceDirection::REVERSE,
            found_cell->get_activations(),
            found_cell->get_activations_alpha(),
            found_cell->get_activations_beta(),
            found_cell->get_clip());
    } else if (ov::is_type<ov::op::v3::GRUCell>(found_cell)) {
        const auto gru_cell = ov::as_type_ptr<ov::op::v3::GRUCell>(found_cell);
        sequence = std::make_shared<ov::op::v5::GRUSequence>(
            X,
            initial_hidden_state,
            seq_lengths,
            W,
            R,
            B,
            gru_cell->get_hidden_size(),
            stride > 0 ? ov::op::RecurrentSequenceDirection::FORWARD : ov::op::RecurrentSequenceDirection::REVERSE,
            gru_cell->get_activations(),
            gru_cell->get_activations_alpha(),
            gru_cell->get_activations_beta(),
            gru_cell->get_clip(),
            gru_cell->get_linear_before_reset());
    } else {
        OPENVINO_THROW("Unsupported sequence type");
    }

    ov::Output<ov::Node> out = sequence->output(0);
    if (slice_axis == 0) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {2, 1, 0, 3});
        out = std::make_shared<ov::op::v1::Transpose>(out, order);
    }

    ov::NodeVector outputs;
    // We must remove num_directions dimension that was added before sequence creation
    auto axis_out = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto out_0 = std::make_shared<ov::op::v0::Squeeze>(out, axis_out);
    auto out_1 = std::make_shared<ov::op::v0::Squeeze>(sequence->output(1), axis_out);
    out_0->set_friendly_name(ti->get_friendly_name() + ".0");
    out_1->set_friendly_name(ti->get_friendly_name() + ".1");
    outputs.emplace_back(out_0);
    outputs.emplace_back(out_1);

    if (sequence->get_output_size() == 3) {
        auto out_2 = std::make_shared<ov::op::v0::Squeeze>(sequence->output(2), axis_out);
        out_2->set_friendly_name(ti->get_friendly_name() + ".2");
        outputs.emplace_back(out_2);
    }

    for (size_t i = 0; i < ordered_out_descs.size(); ++i) {
        if (ordered_out_descs[i]) {
            ti->output(ordered_out_descs[i]->m_output_index).replace(outputs[i]->output(0));
        }
    }

    ov::NodeVector new_nodes = outputs;
    new_nodes.emplace_back(initial_hidden_state);
    new_nodes.emplace_back(W);
    new_nodes.emplace_back(R);
    new_nodes.emplace_back(B);
    new_nodes.emplace_back(sequence);

    if (c_pattern.get_node_shared_ptr()) {
        new_nodes.emplace_back(initial_cell_state);
    }

    new_nodes.emplace_back(batch_dimension);
    new_nodes.emplace_back(shape_of);
    new_nodes.emplace_back(seq_len_dim);
    new_nodes.emplace_back(seq_lengths);

    if (slice_axis == 0) {
        new_nodes.emplace_back(out.get_node_shared_ptr());
        new_nodes.emplace_back(X.get_node_shared_ptr());
    }

    copy_runtime_info(ti, new_nodes);
    return true;
}
}  // namespace

ov::pass::ConvertTensorIteratorToLSTMSequence::ConvertTensorIteratorToLSTMSequence() {
    MATCHER_SCOPE(ConvertTensorIteratorToLSTMSequence);
    auto tensor_iterator = pattern::wrap_type<ov::op::v0::TensorIterator>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(m.get_match_root());
        if (!ti || transformation_callback(ti))
            return false;

        // create a pattern for the TensorIterator body
        auto data = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(3));
        auto pattern_1 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto squeeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Squeeze>({data, pattern_1});

        auto input_H_state = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(2));
        auto input_C_state = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(2));
        auto input_W = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_R = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_B = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));

        ov::OutputVector cell_inputs{squeeze, input_H_state, input_C_state, input_W, input_R, input_B};
        auto cell = ov::pass::pattern::wrap_type<ov::op::v0::LSTMCell, ov::op::v4::LSTMCell>(cell_inputs);

        auto pattern_2 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto unsqueeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({cell, pattern_2});
        ov::pass::pattern::Matcher matcher(unsqueeze);

        bool match = false;
        auto func = ti->get_body();
        for (const auto& res : func->get_results()) {
            match = matcher.match((res->get_input_source_output(0)));
            if (match)
                break;
        }

        // All nodes are in the TI body should be matched in pattern
        if (!match || (matcher.get_matched_nodes().size() + func->get_results().size()) != func->get_ops().size())
            return false;

        const auto& pattern_map = matcher.get_pattern_value_map();
        std::shared_ptr<Node> found_cell = pattern_map.at(cell).get_node_shared_ptr();
        const auto lstm_cell = std::dynamic_pointer_cast<ov::op::util::RNNCellBase>(found_cell);
        if (lstm_cell == nullptr)
            return false;

        return convertTensorIteratorToSequence(ti,
                                               lstm_cell,
                                               pattern_map.at(data),
                                               pattern_map.at(input_H_state),
                                               pattern_map.at(input_C_state),
                                               pattern_map.at(input_W),
                                               pattern_map.at(input_R),
                                               pattern_map.at(input_B),
                                               pattern_map.at(unsqueeze));
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertTensorIteratorToRNNSequence::ConvertTensorIteratorToRNNSequence() {
    MATCHER_SCOPE(ConvertTensorIteratorToRNNSequence);
    auto tensor_iterator = pattern::wrap_type<ov::op::v0::TensorIterator>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(m.get_match_root());
        if (!ti || transformation_callback(ti))
            return false;

        // create a pattern for the TensorIterator body
        auto data = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(3));
        auto pattern_1 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto squeeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Squeeze>({data, pattern_1});

        auto input_H_state = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(2));
        auto input_W = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_R = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_B = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));

        ov::OutputVector cell_inputs{squeeze, input_H_state, input_W, input_R, input_B};
        auto cell = ov::pass::pattern::wrap_type<ov::op::v0::RNNCell>(cell_inputs);

        auto pattern_2 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto unsqueeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({cell, pattern_2});
        ov::pass::pattern::Matcher matcher(unsqueeze);

        bool match = false;
        auto func = ti->get_body();
        for (const auto& res : func->get_results()) {
            match = matcher.match((res->get_input_source_output(0)));
            if (match)
                break;
        }

        // All nodes are in the TI body should be matched in pattern
        if (!match || (matcher.get_matched_nodes().size() + func->get_results().size()) != func->get_ops().size())
            return false;

        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto& rnn_cell =
            std::dynamic_pointer_cast<ov::op::v0::RNNCell>(pattern_map.at(cell).get_node_shared_ptr());
        if (rnn_cell == nullptr)
            return false;

        return convertTensorIteratorToSequence(ti,
                                               rnn_cell,
                                               pattern_map.at(data),
                                               pattern_map.at(input_H_state),
                                               ov::Output<ov::Node>(),
                                               pattern_map.at(input_W),
                                               pattern_map.at(input_R),
                                               pattern_map.at(input_B),
                                               pattern_map.at(unsqueeze));
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertTensorIteratorToGRUSequence::ConvertTensorIteratorToGRUSequence() {
    MATCHER_SCOPE(ConvertTensorIteratorToGRUSequence);
    auto tensor_iterator = pattern::wrap_type<ov::op::v0::TensorIterator>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(m.get_match_root());
        if (!ti || transformation_callback(ti))
            return false;

        // create a pattern for the TensorIterator body
        auto data = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(3));
        auto pattern_1 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto squeeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Squeeze>({data, pattern_1});

        auto input_H_state = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(2));
        auto input_W = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_R = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_B = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));

        ov::OutputVector cell_inputs{squeeze, input_H_state, input_W, input_R, input_B};
        auto cell = ov::pass::pattern::wrap_type<ov::op::v3::GRUCell>(cell_inputs);

        auto pattern_2 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto unsqueeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({cell, pattern_2});

        ov::pass::pattern::Matcher matcher(unsqueeze);

        bool match = false;
        auto func = ti->get_body();
        for (const auto& res : func->get_results()) {
            match = matcher.match((res->get_input_source_output(0)));
            if (match)
                break;
        }

        // All nodes are in the TI body should be matched in pattern
        if (!match || (matcher.get_matched_nodes().size() + func->get_results().size()) != func->get_ops().size())
            return false;

        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto& gru_cell =
            std::dynamic_pointer_cast<ov::op::v3::GRUCell>(pattern_map.at(cell).get_node_shared_ptr());
        if (gru_cell == nullptr)
            return false;

        return convertTensorIteratorToSequence(ti,
                                               gru_cell,
                                               pattern_map.at(data),
                                               pattern_map.at(input_H_state),
                                               ov::Output<ov::Node>(),
                                               pattern_map.at(input_W),
                                               pattern_map.at(input_R),
                                               pattern_map.at(input_B),
                                               pattern_map.at(unsqueeze));
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertTensorIteratorToSequence::ConvertTensorIteratorToSequence() {
    add_matcher<ConvertTensorIteratorToLSTMSequence>();
    add_matcher<ConvertTensorIteratorToRNNSequence>();
    add_matcher<ConvertTensorIteratorToGRUSequence>();
}
