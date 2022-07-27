// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

#include <memory>
#include <ngraph/op/util/activation_functions.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ngraph/builder/autobroadcast.hpp"

namespace {
ngraph::Output<ngraph::Node> get_current_iter(ngraph::ParameterVector& body_params,
                                              ngraph::ResultVector& body_results,
                                              const ngraph::Output<ngraph::Node>& seq_lengths) {
    auto curr_iter_body_param =
        std::make_shared<ngraph::opset5::Parameter>(seq_lengths.get_element_type(), ngraph::Shape{1});
    // increment current iteration
    auto one = ngraph::opset5::Constant::create(seq_lengths.get_element_type(), ngraph::Shape{1}, {1});
    auto add = std::make_shared<ngraph::opset5::Add>(curr_iter_body_param, one);
    auto curr_iter_result = std::make_shared<ngraph::opset5::Result>(add);
    body_params.push_back(curr_iter_body_param);
    body_results.push_back(curr_iter_result);
    return curr_iter_body_param;
}

ngraph::Output<ngraph::Node> get_masked_value(const std::shared_ptr<ngraph::opset5::TensorIterator>& ti,
                                              ngraph::ParameterVector& body_params,
                                              ngraph::ResultVector& body_results,
                                              const ngraph::Output<ngraph::Node>& current_iter,
                                              const ngraph::Output<ngraph::Node>& data,
                                              const ngraph::Output<ngraph::Node>& seq_lengths) {
    // body parameters
    auto aggregated_Y_h_body_param =
        std::make_shared<ngraph::opset5::Parameter>(data.get_element_type(), data.get_partial_shape());

    body_params.push_back(aggregated_Y_h_body_param);

    // Create mask node deciding whether or not to mask batch data.
    auto batch_seq_length = ngraph::builder::opset1::legacy_broadcast_for_binary_operation(data, seq_lengths, 0);

    auto mask_condition = std::make_shared<ngraph::opset5::Greater>(current_iter, batch_seq_length);
    auto mask_Y_h = std::make_shared<ngraph::opset5::Equal>(current_iter, batch_seq_length);

    // Select values depending on mask.
    // Select(<condition>, <true_value>, <false_value>)
    auto select_aggregated_H = std::make_shared<ngraph::opset5::Select>(mask_Y_h, data, aggregated_Y_h_body_param);
    auto aggregated_result = std::make_shared<ngraph::opset5::Result>(select_aggregated_H);
    body_results.push_back(aggregated_result);

    auto scalar_mask_value = ngraph::opset5::Constant::create(data.get_element_type(), {}, {0.f});
    auto data_shape = ngraph::op::util::make_try_fold<ngraph::opset5::ShapeOf>(data);
    auto mask_value = ngraph::op::util::make_try_fold<ngraph::opset5::Broadcast>(scalar_mask_value, data_shape);
    return ngraph::op::util::make_try_fold<ngraph::opset5::Select>(mask_condition, mask_value, data);
}

bool convert_sequence_to_ti(const std::shared_ptr<ngraph::Node>& sequence,
                            const ngraph::Output<ngraph::Node>& X,
                            const ngraph::Output<ngraph::Node>& H_t,
                            const ngraph::Output<ngraph::Node>& C_t,
                            const ngraph::Output<ngraph::Node>& seq_lengths,
                            const ngraph::Output<ngraph::Node>& W,
                            const ngraph::Output<ngraph::Node>& R,
                            const ngraph::Output<ngraph::Node>& B,
                            const ngraph::op::RecurrentSequenceDirection& direction) {
    auto X_pshape = X.get_partial_shape();
    if (X_pshape.size() < 2 || X_pshape[1].is_dynamic()) {
        return false;
    }

    auto max_seq_len = X_pshape[1].get_length();
    bool enable_mask = ngraph::op::util::is_seq_len_provided(seq_lengths.get_node_shared_ptr(), max_seq_len);

    const bool is_reverse = direction == ngraph::op::RecurrentSequenceDirection::REVERSE;
    std::shared_ptr<ngraph::Node> reverse_seq_before;
    if (is_reverse && enable_mask) {
        reverse_seq_before = std::make_shared<ngraph::opset5::ReverseSequence>(X, seq_lengths, 0, 1);
    }

    auto axis_0 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
    auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});

    // TensorIterator Body: begin
    auto X_param_pshape = X_pshape;
    X_param_pshape[1] = 1;  // split by seq_lengths dimension
    auto X_body_param = std::make_shared<ngraph::opset5::Parameter>(X.get_element_type(), X_param_pshape);

    const auto squeezed_h = ngraph::op::util::make_try_fold<ngraph::opset5::Squeeze>(H_t, axis_1);
    auto H_body_param = std::make_shared<ngraph::opset5::Parameter>(squeezed_h->get_element_type(),
                                                                    squeezed_h->get_output_partial_shape(0));
    auto seq_body_param =
        std::make_shared<ngraph::opset5::Parameter>(seq_lengths.get_element_type(), seq_lengths.get_partial_shape());

    // LSTM sequence case
    const bool cell_state_defined = C_t.get_node_shared_ptr() != nullptr;
    std::shared_ptr<ngraph::opset5::Parameter> C_body_param = nullptr;
    std::shared_ptr<ngraph::Node> squeezed_c = nullptr;
    if (cell_state_defined) {
        squeezed_c = ngraph::op::util::make_try_fold<ngraph::opset5::Squeeze>(C_t, axis_1);
        C_body_param = std::make_shared<ngraph::opset5::Parameter>(squeezed_c->get_element_type(),
                                                                   squeezed_c->get_output_partial_shape(0));
    }

    const auto squeezed_x = ngraph::op::util::make_try_fold<ngraph::opset5::Squeeze>(X_body_param, axis_1);
    const auto squeezed_w = ngraph::op::util::make_try_fold<ngraph::opset5::Squeeze>(W, axis_0);
    const auto squeezed_r = ngraph::op::util::make_try_fold<ngraph::opset5::Squeeze>(R, axis_0);
    const auto squeezed_b = ngraph::op::util::make_try_fold<ngraph::opset5::Squeeze>(B, axis_0);

    std::shared_ptr<ngraph::Node> cell;
    if (const auto lstm_sequence = ngraph::as_type_ptr<ngraph::opset5::LSTMSequence>(sequence)) {
        cell = std::make_shared<ngraph::opset5::LSTMCell>(squeezed_x,
                                                          H_body_param,
                                                          C_body_param,
                                                          squeezed_w,
                                                          squeezed_r,
                                                          squeezed_b,
                                                          lstm_sequence->get_hidden_size(),
                                                          lstm_sequence->get_activations(),
                                                          lstm_sequence->get_activations_alpha(),
                                                          lstm_sequence->get_activations_beta(),
                                                          lstm_sequence->get_clip());
    } else if (const auto rnn_sequence = ngraph::as_type_ptr<ngraph::opset5::RNNSequence>(sequence)) {
        cell = std::make_shared<ngraph::opset5::RNNCell>(squeezed_x,
                                                         H_body_param,
                                                         squeezed_w,
                                                         squeezed_r,
                                                         squeezed_b,
                                                         rnn_sequence->get_hidden_size(),
                                                         rnn_sequence->get_activations(),
                                                         rnn_sequence->get_activations_alpha(),
                                                         rnn_sequence->get_activations_beta(),
                                                         rnn_sequence->get_clip());
    } else if (const auto gnn_sequence = ngraph::as_type_ptr<ngraph::opset5::GRUSequence>(sequence)) {
        cell = std::make_shared<ngraph::opset5::GRUCell>(squeezed_x,
                                                         H_body_param,
                                                         squeezed_w,
                                                         squeezed_r,
                                                         squeezed_b,
                                                         gnn_sequence->get_hidden_size(),
                                                         gnn_sequence->get_activations(),
                                                         gnn_sequence->get_activations_alpha(),
                                                         gnn_sequence->get_activations_beta(),
                                                         gnn_sequence->get_clip(),
                                                         gnn_sequence->get_linear_before_reset());
    } else {
        return false;
    }

    ngraph::ParameterVector body_params;
    ngraph::ResultVector body_results;

    ngraph::Output<ngraph::Node> hidden_state = cell->output(0);
    ngraph::Output<ngraph::Node> cell_state;
    if (cell_state_defined)
        cell_state = cell->output(1);

    auto tensor_iterator = std::make_shared<ngraph::opset5::TensorIterator>();
    if (enable_mask) {
        const auto current_iter = get_current_iter(body_params, body_results, seq_body_param);
        hidden_state =
            get_masked_value(tensor_iterator, body_params, body_results, current_iter, hidden_state, seq_body_param);
        if (cell_state_defined)
            cell_state =
                get_masked_value(tensor_iterator, body_params, body_results, current_iter, cell_state, seq_body_param);
    }

    auto H_res = std::make_shared<ngraph::opset5::Result>(hidden_state);
    auto C_res = cell_state_defined ? std::make_shared<ngraph::opset5::Result>(cell_state) : nullptr;
    auto hidden_state_unsqueezed = std::make_shared<ngraph::opset5::Unsqueeze>(hidden_state, axis_1);
    auto concat_res = std::make_shared<ngraph::opset5::Result>(hidden_state_unsqueezed);

    body_params.push_back(X_body_param);
    body_params.push_back(H_body_param);
    if (cell_state_defined)
        body_params.push_back(C_body_param);
    body_params.push_back(seq_body_param);

    body_results.push_back(concat_res);
    body_results.push_back(H_res);
    if (cell_state_defined)
        body_results.push_back(C_res);

    auto body = std::make_shared<ngraph::Function>(body_results, body_params);
    tensor_iterator->set_function(body);
    // TensorIterator Body: end
    if (is_reverse) {
        if (!enable_mask) {
            // Reversed order, stride -1
            tensor_iterator->set_sliced_input(X_body_param, X, -1, -1, 1, 0, 1);
            tensor_iterator->get_concatenated_slices(concat_res, -1, -1, 1, 0, 1);
        } else {
            // use ReverseSequence as initializer
            tensor_iterator->set_sliced_input(X_body_param, reverse_seq_before, 0, 1, 1, -1, 1);
            tensor_iterator->get_concatenated_slices(concat_res, 0, 1, 1, -1, 1);
        }
    } else {
        // forward order
        tensor_iterator->set_sliced_input(X_body_param, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(concat_res, 0, 1, 1, -1, 1);
    }

    tensor_iterator->set_merged_input(H_body_param, squeezed_h, H_res);
    if (cell_state_defined)
        tensor_iterator->set_merged_input(C_body_param, squeezed_c, C_res);
    tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

    ngraph::Output<ngraph::Node> H_out = H_res;
    ngraph::Output<ngraph::Node> C_out = C_res;
    if (enable_mask) {
        // create initial values for body_parameters in outer graph
        // aggregated Y_h - concatenation of the last non-zero values for each batch
        auto H_body_param_shape = ngraph::op::util::make_try_fold<ngraph::opset5::ShapeOf>(H_body_param);
        auto aggregated_Y_h_scalar = ngraph::opset5::Constant::create(H_body_param->get_element_type(), {}, {0.f});
        auto aggregated_Y_h =
            ngraph::op::util::make_try_fold<ngraph::opset5::Broadcast>(aggregated_Y_h_scalar, H_body_param_shape);

        auto init_val_curr_iter =
            ngraph::opset5::Constant::create(seq_lengths.get_element_type(), ngraph::Shape{1}, {1});
        ngraph::copy_runtime_info(sequence, {aggregated_Y_h, init_val_curr_iter});

        // set initial value and back edge for current iteration
        tensor_iterator->set_merged_input(body_params.at(0), init_val_curr_iter, body_results.at(0));
        // set initial value and back edge for aggregated H
        tensor_iterator->set_merged_input(body_params.at(1), aggregated_Y_h, body_results.at(1));

        H_out = tensor_iterator->get_function()->get_results()[1];

        if (cell_state_defined) {
            auto C_body_param_shape = ngraph::op::util::make_try_fold<ngraph::opset5::ShapeOf>(C_body_param);
            auto aggregated_Y_c_scalar = ngraph::opset5::Constant::create(C_body_param->get_element_type(), {}, {0.f});
            auto aggregated_Y_c =
                ngraph::op::util::make_try_fold<ngraph::opset5::Broadcast>(aggregated_Y_c_scalar, C_body_param_shape);
            ngraph::copy_runtime_info(sequence, aggregated_Y_c);

            // set initial value and back edge for aggregated C
            tensor_iterator->set_merged_input(body_params.at(2), aggregated_Y_c, body_results.at(2));
            C_out = tensor_iterator->get_function()->get_results()[2];
        }
    }

    tensor_iterator->get_iter_value(H_out);
    if (cell_state_defined)
        tensor_iterator->get_iter_value(C_out);
    tensor_iterator->set_friendly_name(sequence->get_friendly_name());
    ngraph::NodeVector new_nodes{squeezed_h, tensor_iterator};
    if (cell_state_defined)
        new_nodes.push_back(squeezed_c);
    ngraph::OutputVector nodes_to_replace;
    if (enable_mask && is_reverse) {
        auto reverse_seq_after =
            std::make_shared<ngraph::opset5::ReverseSequence>(tensor_iterator->output(0), seq_lengths, 0, 1);
        // Resolve a collision of names data nodes in CNN Network in Reverse case with mask.
        /*
         *   Before transformation (no collisions)
         *   RNN/LSTM/GRU Sequence [rnn_name] -- (data_node: rnn_name.0) - > Result1
         *                                    -- (data_node: rnn_name.1) - > Result2
         *
         *
         *   After transformation (without identity, there are collisions):
         *   We need to set rnn_name.0 to RevSequence to store result name.
         *   TI [rnn_name] -- (DATA_NODE: rnn_name.0) --> RevSequence [rnn_name.0] -- (DATA_NODE: rnn_name.0) -> Result1
         *                 -- (data_node: rnn_name.1) --> Result2
         *
         *
         *   After transformation (with identity, no collisions):
         *   TI has other_name, but it doesn't affect result names due TI is not connected to Results directly.
         *   TI [other_name] -- (data_node: other_name.0) --> RevSequence [rnn_name.0] -- (data_node: rnn_name.0) ->
         * Result1
         *                   -- (data_node: other_name.1) --> Identity(rnn_name.1) -- (data_node: rnn_name.1) -> Result2
         */
        new_nodes.push_back(reverse_seq_before);
        new_nodes.push_back(reverse_seq_after);
        nodes_to_replace.push_back(reverse_seq_after);
        nodes_to_replace.push_back(tensor_iterator->output(1));

        if (cell_state_defined) {
            auto cell_state = tensor_iterator->output(2);
            new_nodes.emplace_back(cell_state.get_node_shared_ptr());
            nodes_to_replace.emplace_back(cell_state);
        }

        tensor_iterator->set_friendly_name(sequence->get_friendly_name() + "/tensor_iterator");
    } else {
        nodes_to_replace = tensor_iterator->outputs();
    }

    for (size_t i = 0; i < nodes_to_replace.size(); i++) {
        auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(nodes_to_replace[i], axis_1);
        unsqueeze->set_friendly_name(sequence->get_friendly_name() + "." + std::to_string(i));
        nodes_to_replace[i] = unsqueeze;
        new_nodes.push_back(unsqueeze);
    }
    ngraph::copy_runtime_info(sequence, new_nodes);
    ngraph::replace_node(sequence, nodes_to_replace);

    return true;
}
}  // namespace

ngraph::pass::ConvertRNNSequenceToTensorIterator::ConvertRNNSequenceToTensorIterator() {
    MATCHER_SCOPE(ConvertRNNSequenceToTensorIterator);
    auto X_m = pattern::any_input(pattern::has_static_rank());
    auto H_t_m = pattern::any_input();
    auto seq_lengths_m = pattern::any_input();
    auto W_m = pattern::any_input();
    auto R_m = pattern::any_input();
    auto B_m = pattern::any_input();
    auto rnn_seq = ngraph::pattern::wrap_type<opset5::RNNSequence>({X_m, H_t_m, seq_lengths_m, W_m, R_m, B_m});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto sequence = ngraph::as_type_ptr<ngraph::opset5::RNNSequence>(m.get_match_root());

        // Bidirectional Sequence op should be decomposed to Reverse + Forward
        // (e.g. apply BidirectionalRNNSequenceDecomposition transformation before this one)
        if (!sequence || sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ||
            transformation_callback(sequence)) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        const auto& X = pattern_map.at(X_m);                      // split
        const auto& H_t = pattern_map.at(H_t_m);                  // merged (init value + back edge)
        const auto& seq_lengths = pattern_map.at(seq_lengths_m);  // invariant
        const auto& W = pattern_map.at(W_m);                      // const in the body
        const auto& R = pattern_map.at(R_m);                      // const in the body
        const auto& B = pattern_map.at(B_m);                      // const in the body

        return convert_sequence_to_ti(sequence,
                                      X,
                                      H_t,
                                      Output<Node>(),
                                      seq_lengths,
                                      W,
                                      R,
                                      B,
                                      sequence->get_direction());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_seq, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::ConvertGRUSequenceToTensorIterator::ConvertGRUSequenceToTensorIterator() {
    MATCHER_SCOPE(ConvertGRUSequenceToTensorIterator);
    auto X_m = pattern::any_input(pattern::has_static_rank());
    auto H_t_m = pattern::any_input();
    auto seq_lengths_m = pattern::any_input();
    auto W_m = pattern::any_input();
    auto R_m = pattern::any_input();
    auto B_m = pattern::any_input();
    auto gru_seq = ngraph::pattern::wrap_type<opset5::GRUSequence>({X_m, H_t_m, seq_lengths_m, W_m, R_m, B_m});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto sequence = ngraph::as_type_ptr<ngraph::opset5::GRUSequence>(m.get_match_root());

        // Bidirectional Sequence op should be decomposed to Reverse + Forward
        // (e.g. apply BidirectionalRNNSequenceDecomposition transformation before this one)
        if (!sequence || sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ||
            transformation_callback(sequence)) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        const auto& X = pattern_map.at(X_m);                      // split
        const auto& H_t = pattern_map.at(H_t_m);                  // merged (init value + back edge)
        const auto& seq_lengths = pattern_map.at(seq_lengths_m);  // invariant
        const auto& W = pattern_map.at(W_m);                      // const in the body
        const auto& R = pattern_map.at(R_m);                      // const in the body
        const auto& B = pattern_map.at(B_m);                      // const in the body

        return convert_sequence_to_ti(sequence,
                                      X,
                                      H_t,
                                      Output<Node>(),
                                      seq_lengths,
                                      W,
                                      R,
                                      B,
                                      sequence->get_direction());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gru_seq, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::ConvertLSTMSequenceToTensorIterator::ConvertLSTMSequenceToTensorIterator() {
    MATCHER_SCOPE(ConvertLSTMSequenceToTensorIterator);
    auto X_m = pattern::any_input(pattern::has_static_rank());
    auto H_t_m = pattern::any_input();
    auto C_t_m = pattern::any_input();
    auto seq_lengths_m = pattern::any_input();
    auto W_m = pattern::any_input();
    auto R_m = pattern::any_input();
    auto B_m = pattern::any_input();
    auto lstm_seq = ngraph::pattern::wrap_type<opset5::LSTMSequence>({X_m, H_t_m, C_t_m, seq_lengths_m, W_m, R_m, B_m});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto sequence = ngraph::as_type_ptr<ngraph::opset5::LSTMSequence>(m.get_match_root());

        // Bidirectional Sequence op should be decomposed to Reverse + Forward
        // (e.g. apply BidirectionalRNNSequenceDecomposition transformation before this one)
        if (!sequence || sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ||
            transformation_callback(sequence)) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        const auto& X = pattern_map.at(X_m);                      // split
        const auto& H_t = pattern_map.at(H_t_m);                  // merged (init value + back edge)
        const auto& C_t = pattern_map.at(C_t_m);                  // merged (init value + back edge)
        const auto& seq_lengths = pattern_map.at(seq_lengths_m);  // invariant
        const auto& W = pattern_map.at(W_m);                      // const in the body
        const auto& R = pattern_map.at(R_m);                      // const in the body
        const auto& B = pattern_map.at(B_m);                      // const in the body

        return convert_sequence_to_ti(sequence, X, H_t, C_t, seq_lengths, W, R, B, sequence->get_direction());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstm_seq, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::ConvertSequenceToTensorIterator::ConvertSequenceToTensorIterator() {
    add_matcher<ConvertLSTMSequenceToTensorIterator>();
    add_matcher<ConvertRNNSequenceToTensorIterator>();
    add_matcher<ConvertGRUSequenceToTensorIterator>();
}
