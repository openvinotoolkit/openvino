// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_ti_to_sequences.hpp"

#include <memory>
#include <ngraph/graph_util.hpp>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

namespace {
bool convertTensorIteratorToSequence(const std::shared_ptr<ngraph::opset5::TensorIterator>& ti,
                                     const std::shared_ptr<ngraph::op::util::RNNCellBase>& found_cell,
                                     const ngraph::Output<ngraph::Node>& data,
                                     const ngraph::Output<ngraph::Node>& h_pattern,
                                     const ngraph::Output<ngraph::Node>& c_pattern,
                                     const ngraph::Output<ngraph::Node>& w_pattern,
                                     const ngraph::Output<ngraph::Node>& r_pattern,
                                     const ngraph::Output<ngraph::Node>& b_pattern,
                                     const ngraph::Output<ngraph::Node>& unsqueeze_after_cell) {
    const auto& func = ti->get_function();
    const auto& params = func->get_parameters();

    std::vector<std::shared_ptr<ngraph::opset5::TensorIterator::InputDescription>> ordered_in_descs(3);
    int64_t stride = 0, slice_axis = 0;

    // Remember the order of the X and initial_hidden_state (+ initial_cell_state in case of LSTM) in the TensorIterator
    // params
    for (const auto& input_desc : ti->get_input_descriptions()) {
        auto param = params[input_desc->m_body_parameter_index];
        if (param == data.get_node_shared_ptr()) {
            auto slice_input =
                std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::SliceInputDescription>(input_desc);
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
    std::vector<std::shared_ptr<ngraph::opset5::TensorIterator::OutputDescription>> ordered_out_descs(3);

    // Remember the order of cell outputs in the TensorIterator
    for (const auto& output_desc : ti->get_output_descriptions()) {
        std::shared_ptr<ngraph::opset5::Result> res = results[output_desc->m_body_value_index];
        if (res->input_value(0) == unsqueeze_after_cell) {
            auto concat_output =
                std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::ConcatOutputDescription>(output_desc);
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
        auto order = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 0, 2});
        X = std::make_shared<ngraph::opset5::Transpose>(ti_inputs[ordered_in_descs[0]->m_input_index], order);
    }

    // We must prepare cell inputs to sequence creation: insert num_directions elem via unsqueeze where needed (please,
    // see specification)
    auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
    auto initial_hidden_state =
        std::make_shared<ngraph::opset5::Unsqueeze>(ti_inputs[ordered_in_descs[1]->m_input_index], axis_1);

    // LSTM case
    std::shared_ptr<ngraph::Node> initial_cell_state =
        c_pattern.get_node_shared_ptr() == nullptr
            ? nullptr
            : std::make_shared<ngraph::opset5::Unsqueeze>(ti_inputs[ordered_in_descs[2]->m_input_index], axis_1);

    const size_t batch_dim = slice_axis == 0 ? 1 : 0;
    auto batch_dimension = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_source(
        ti_inputs[ordered_in_descs[0]->m_input_index],
        {batch_dim});

    auto seq_lengths_scalar = ngraph::opset5::Constant::create(ngraph::element::i32, {}, {ti->get_num_iterations()});
    auto seq_lengths = ngraph::op::util::make_try_fold<ngraph::opset5::Broadcast>(seq_lengths_scalar, batch_dimension);

    auto axis_0 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
    auto W = ngraph::op::util::make_try_fold<ngraph::opset5::Unsqueeze>(w_pattern, axis_0);
    auto R = ngraph::op::util::make_try_fold<ngraph::opset5::Unsqueeze>(r_pattern, axis_0);
    auto B = ngraph::op::util::make_try_fold<ngraph::opset5::Unsqueeze>(b_pattern, axis_0);

    std::shared_ptr<ngraph::Node> sequence;
    if (ngraph::is_type<ngraph::opset5::LSTMCell>(found_cell) ||
        ngraph::is_type<ngraph::opset1::LSTMCell>(found_cell)) {
        sequence =
            std::make_shared<ngraph::opset5::LSTMSequence>(X,
                                                           initial_hidden_state,
                                                           initial_cell_state,
                                                           seq_lengths,
                                                           W,
                                                           R,
                                                           B,
                                                           found_cell->get_hidden_size(),
                                                           stride > 0 ? ngraph::op::RecurrentSequenceDirection::FORWARD
                                                                      : ngraph::op::RecurrentSequenceDirection::REVERSE,
                                                           found_cell->get_activations_alpha(),
                                                           found_cell->get_activations_beta(),
                                                           found_cell->get_activations(),
                                                           found_cell->get_clip());
    } else if (ngraph::is_type<ngraph::opset5::RNNCell>(found_cell)) {
        sequence =
            std::make_shared<ngraph::opset5::RNNSequence>(X,
                                                          initial_hidden_state,
                                                          seq_lengths,
                                                          W,
                                                          R,
                                                          B,
                                                          found_cell->get_hidden_size(),
                                                          stride > 0 ? ngraph::op::RecurrentSequenceDirection::FORWARD
                                                                     : ngraph::op::RecurrentSequenceDirection::REVERSE,
                                                          found_cell->get_activations(),
                                                          found_cell->get_activations_alpha(),
                                                          found_cell->get_activations_beta(),
                                                          found_cell->get_clip());
    } else if (ngraph::is_type<ngraph::opset5::GRUCell>(found_cell)) {
        const auto gru_cell = ngraph::as_type_ptr<ngraph::opset5::GRUCell>(found_cell);
        sequence =
            std::make_shared<ngraph::opset5::GRUSequence>(X,
                                                          initial_hidden_state,
                                                          seq_lengths,
                                                          W,
                                                          R,
                                                          B,
                                                          gru_cell->get_hidden_size(),
                                                          stride > 0 ? ngraph::op::RecurrentSequenceDirection::FORWARD
                                                                     : ngraph::op::RecurrentSequenceDirection::REVERSE,
                                                          gru_cell->get_activations(),
                                                          gru_cell->get_activations_alpha(),
                                                          gru_cell->get_activations_beta(),
                                                          gru_cell->get_clip(),
                                                          gru_cell->get_linear_before_reset());
    } else {
        throw ngraph::ngraph_error("Unsupported sequence type");
    }

    ngraph::Output<ngraph::Node> out = sequence->output(0);
    if (slice_axis == 0) {
        auto order = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 1, 0, 3});
        out = std::make_shared<ngraph::opset5::Transpose>(out, order);
    }

    ngraph::NodeVector outputs;
    // We must remove num_directions dimension that was added before sequence creation
    auto axis_out = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
    auto out_0 = std::make_shared<ngraph::opset5::Squeeze>(out, axis_out);
    auto out_1 = std::make_shared<ngraph::opset5::Squeeze>(sequence->output(1), axis_out);
    out_0->set_friendly_name(ti->get_friendly_name() + ".0");
    out_1->set_friendly_name(ti->get_friendly_name() + ".1");
    outputs.emplace_back(out_0);
    outputs.emplace_back(out_1);

    if (sequence->get_output_size() == 3) {
        auto out_2 = std::make_shared<ngraph::opset5::Squeeze>(sequence->output(2), axis_out);
        out_2->set_friendly_name(ti->get_friendly_name() + ".2");
        outputs.emplace_back(out_2);
    }

    for (size_t i = 0; i < ordered_out_descs.size(); ++i) {
        if (ordered_out_descs[i]) {
            for (const auto& input : ti->output(ordered_out_descs[i]->m_output_index).get_target_inputs()) {
                input.replace_source_output(outputs[i]->output(0));
            }
            NGRAPH_SUPPRESS_DEPRECATED_START
            outputs[i]->get_output_tensor(0).set_name(
                ngraph::op::util::create_ie_output_name(ti->output(ordered_out_descs[i]->m_output_index)));
            NGRAPH_SUPPRESS_DEPRECATED_END
        }
    }

    ngraph::NodeVector new_nodes = outputs;
    new_nodes.emplace_back(initial_hidden_state);
    new_nodes.emplace_back(W);
    new_nodes.emplace_back(R);
    new_nodes.emplace_back(B);
    new_nodes.emplace_back(sequence);

    if (c_pattern.get_node_shared_ptr()) {
        new_nodes.emplace_back(initial_cell_state);
    }
    if (!std::dynamic_pointer_cast<ngraph::opset5::Constant>(seq_lengths)) {
        new_nodes.emplace_back(batch_dimension);
        new_nodes.emplace_back(batch_dimension->get_input_node_shared_ptr(0));
        new_nodes.emplace_back(seq_lengths_scalar);
        new_nodes.emplace_back(seq_lengths);
    }
    if (slice_axis == 0) {
        new_nodes.emplace_back(out.get_node_shared_ptr());
        new_nodes.emplace_back(X.get_node_shared_ptr());
    }

    copy_runtime_info(ti, new_nodes);
    return true;
}
}  // namespace

ngraph::pass::ConvertTensorIteratorToLSTMSequence::ConvertTensorIteratorToLSTMSequence() {
    MATCHER_SCOPE(ConvertTensorIteratorToLSTMSequence);
    auto tensor_iterator = pattern::wrap_type<ngraph::opset5::TensorIterator>();

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator>(m.get_match_root());
        if (!ti || transformation_callback(ti))
            return false;

        // create a pattern for the TensorIterator body
        auto data = ngraph::pattern::wrap_type<ngraph::opset5::Parameter>(ngraph::pattern::rank_equals(3));
        auto pattern_1 = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(1));
        auto squeeze = ngraph::pattern::wrap_type<ngraph::opset5::Reshape, ngraph::opset5::Squeeze>({data, pattern_1});

        auto input_H_state = ngraph::pattern::wrap_type<ngraph::opset5::Parameter>(ngraph::pattern::rank_equals(2));
        auto input_C_state = ngraph::pattern::wrap_type<ngraph::opset5::Parameter>(ngraph::pattern::rank_equals(2));
        auto input_W = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(2));
        auto input_R = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(2));
        auto input_B = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(1));

        ngraph::OutputVector cell_inputs{squeeze, input_H_state, input_C_state, input_W, input_R, input_B};
        auto cell = ngraph::pattern::wrap_type<ngraph::opset1::LSTMCell, ngraph::opset5::LSTMCell>(cell_inputs);

        auto pattern_2 = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(1));
        auto unsqueeze = ngraph::pattern::wrap_type<ngraph::opset5::Reshape>({cell, pattern_2});
        ngraph::pattern::Matcher matcher(unsqueeze);

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
        const auto lstm_cell = std::dynamic_pointer_cast<ngraph::op::util::RNNCellBase>(found_cell);
        if (lstm_cell == nullptr)
            return false;

        MATCHER_SCOPE_ENABLE(ConvertTensorIteratorToLSTMSequence);
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

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::ConvertTensorIteratorToRNNSequence::ConvertTensorIteratorToRNNSequence() {
    MATCHER_SCOPE(ConvertTensorIteratorToRNNSequence);
    auto tensor_iterator = pattern::wrap_type<ngraph::opset5::TensorIterator>();

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator>(m.get_match_root());
        if (!ti || transformation_callback(ti))
            return false;

        // create a pattern for the TensorIterator body
        auto data = ngraph::pattern::wrap_type<ngraph::opset5::Parameter>(ngraph::pattern::rank_equals(3));
        auto pattern_1 = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(1));
        auto squeeze = ngraph::pattern::wrap_type<ngraph::opset5::Reshape, ngraph::opset5::Squeeze>({data, pattern_1});

        auto input_H_state = ngraph::pattern::wrap_type<ngraph::opset5::Parameter>(ngraph::pattern::rank_equals(2));
        auto input_W = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(2));
        auto input_R = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(2));
        auto input_B = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(1));

        ngraph::OutputVector cell_inputs{squeeze, input_H_state, input_W, input_R, input_B};
        auto cell = ngraph::pattern::wrap_type<ngraph::opset5::RNNCell>(cell_inputs);

        auto pattern_2 = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(1));
        auto unsqueeze = ngraph::pattern::wrap_type<ngraph::opset5::Reshape>({cell, pattern_2});
        ngraph::pattern::Matcher matcher(unsqueeze);

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
            std::dynamic_pointer_cast<ngraph::opset5::RNNCell>(pattern_map.at(cell).get_node_shared_ptr());
        if (rnn_cell == nullptr)
            return false;
        MATCHER_SCOPE_ENABLE(ConvertTensorIteratorToRNNSequence);
        return convertTensorIteratorToSequence(ti,
                                               rnn_cell,
                                               pattern_map.at(data),
                                               pattern_map.at(input_H_state),
                                               ngraph::Output<ngraph::Node>(),
                                               pattern_map.at(input_W),
                                               pattern_map.at(input_R),
                                               pattern_map.at(input_B),
                                               pattern_map.at(unsqueeze));
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::ConvertTensorIteratorToGRUSequence::ConvertTensorIteratorToGRUSequence() {
    MATCHER_SCOPE(ConvertTensorIteratorToGRUSequence);
    auto tensor_iterator = pattern::wrap_type<ngraph::opset5::TensorIterator>();

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator>(m.get_match_root());
        if (!ti || transformation_callback(ti))
            return false;

        // create a pattern for the TensorIterator body
        auto data = ngraph::pattern::wrap_type<ngraph::opset5::Parameter>(ngraph::pattern::rank_equals(3));
        auto pattern_1 = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(1));
        auto squeeze = ngraph::pattern::wrap_type<ngraph::opset5::Reshape, ngraph::opset5::Squeeze>({data, pattern_1});

        auto input_H_state = ngraph::pattern::wrap_type<ngraph::opset5::Parameter>(ngraph::pattern::rank_equals(2));
        auto input_W = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(2));
        auto input_R = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(2));
        auto input_B = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(1));

        ngraph::OutputVector cell_inputs{squeeze, input_H_state, input_W, input_R, input_B};
        auto cell = ngraph::pattern::wrap_type<ngraph::opset5::GRUCell>(cell_inputs);

        auto pattern_2 = ngraph::pattern::wrap_type<ngraph::opset5::Constant>(ngraph::pattern::rank_equals(1));
        auto unsqueeze = ngraph::pattern::wrap_type<ngraph::opset5::Reshape>({cell, pattern_2});
        ngraph::pattern::Matcher matcher(unsqueeze);

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
            std::dynamic_pointer_cast<ngraph::opset5::GRUCell>(pattern_map.at(cell).get_node_shared_ptr());
        if (gru_cell == nullptr)
            return false;
        MATCHER_SCOPE_ENABLE(ConvertTensorIteratorToGRUSequence);
        return convertTensorIteratorToSequence(ti,
                                               gru_cell,
                                               pattern_map.at(data),
                                               pattern_map.at(input_H_state),
                                               ngraph::Output<ngraph::Node>(),
                                               pattern_map.at(input_W),
                                               pattern_map.at(input_R),
                                               pattern_map.at(input_B),
                                               pattern_map.at(unsqueeze));
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::ConvertTensorIteratorToSequence::ConvertTensorIteratorToSequence() {
    add_matcher<ConvertTensorIteratorToLSTMSequence>();
    add_matcher<ConvertTensorIteratorToRNNSequence>();
    add_matcher<ConvertTensorIteratorToGRUSequence>();
}
