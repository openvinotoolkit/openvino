// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ngraph/builder/autobroadcast.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

#include <memory>
#include <transformations/utils/utils.hpp>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/op/util/activation_functions.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertRNNSequenceToTensorIterator, "ConvertRNNSequenceToTensorIterator", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertGRUSequenceToTensorIterator, "ConvertGRUSequenceToTensorIterator", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertLSTMSequenceToTensorIterator, "ConvertLSTMSequenceToTensorIterator", 0);

ngraph::Output<ngraph::Node> get_current_iter(ngraph::ParameterVector& body_params,
                                              ngraph::ResultVector& body_results,
                                              const ngraph::Output<ngraph::Node>& seq_lengths) {
    auto curr_iter_body_param = std::make_shared<ngraph::opset5::Parameter>(seq_lengths.get_element_type(),
                                                                            ngraph::Shape{1});
    // increment current iteration
    auto one = std::make_shared<ngraph::opset5::Constant>(seq_lengths.get_element_type(), ngraph::Shape{1},
                                                          std::vector<int64_t>{1});
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
    const auto& data_type = data.get_element_type();
    const auto& data_shape = data.get_shape();
    const auto& data_shape_size = ngraph::shape_size(data_shape);

    // body parameters
    auto aggregated_Y_h_body_param = std::make_shared<ngraph::opset5::Parameter>(data_type, data_shape);

    body_params.push_back(aggregated_Y_h_body_param);

    // Create mask node deciding whether or not to mask batch data.
    ngraph::Output<ngraph::Node> batch_seq_length = ngraph::builder::opset1::legacy_broadcast_for_binary_operation(
            data, seq_lengths, 0);
    auto mask_value = std::make_shared<ngraph::opset5::Constant>(data.get_element_type(),
                             data.get_shape(), std::vector<float>(data_shape_size, 0.f));
    auto mask_condition = std::make_shared<ngraph::opset5::Greater>(current_iter, batch_seq_length);
    auto mask_Y_h = std::make_shared<ngraph::opset5::Equal>(current_iter, batch_seq_length);

    // Select values depending on mask.
    // Select(<condition>, <true_value>, <false_value>)
    auto select_aggregated_H = std::make_shared<ngraph::opset5::Select>(mask_Y_h, data, aggregated_Y_h_body_param);
    auto aggregated_result = std::make_shared<ngraph::opset5::Result>(select_aggregated_H);
    body_results.push_back(aggregated_result);
    return std::make_shared<ngraph::opset5::Select>(mask_condition, mask_value, data);
}

ngraph::NodeVector squeeze_nodes(const ngraph::OutputVector& nodes_to_squeeze, const ngraph::OutputVector & axes) {
    ngraph::NodeVector squeezed_nodes(nodes_to_squeeze.size());
    for (size_t i = 0; i < nodes_to_squeeze.size(); ++i) {
        squeezed_nodes[i] = std::make_shared<ngraph::opset5::Squeeze>(nodes_to_squeeze[i], axes[i]);
    }
    return squeezed_nodes;
}

bool should_enable_mask(const ngraph::Output<ngraph::Node>& seq_lengths, size_t max_seq_len) {
    bool enable_mask = true;
    // disable the mask if all values of seq_lengths input are equal to max_seq_len (X_shape[1])
    if (const auto &seq_len_const = std::dynamic_pointer_cast<ngraph::opset5::Constant>(
            seq_lengths.get_node_shared_ptr())) {
        const auto &seq_len_values = seq_len_const->cast_vector<int64_t>();
        for (const auto &val : seq_len_values) {
            enable_mask &= (val == max_seq_len);
        }
        enable_mask = !enable_mask;
    }
    return enable_mask;
}

ngraph::pass::ConvertRNNSequenceToTensorIterator::ConvertRNNSequenceToTensorIterator() {
    auto rnn_seq = ngraph::pattern::wrap_type<opset5::RNNSequence>();
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto rnn_seq = std::dynamic_pointer_cast<ngraph::opset5::RNNSequence>(m.get_match_root());

        // Bidirectional Sequence op should be decomposed to Reverse + Forward
        // (e.g. apply BidirectionalRNNSequenceDecomposition transformation before this one)
        if (!rnn_seq && rnn_seq->get_direction() != ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
            return false;
        }

        NodeVector new_nodes;
        const auto &X = rnn_seq->input_value(0); // split
        const auto &H_t = rnn_seq->input_value(1); // merged (init value + back edge)
        const auto &seq_lengths = rnn_seq->input_value(2); // invariant
        const auto &W = rnn_seq->input_value(3); // const in the body
        const auto &R = rnn_seq->input_value(4); // const in the body
        const auto &B = rnn_seq->input_value(5); // const in the body
        bool is_reverse = rnn_seq->get_direction() == ngraph::op::RecurrentSequenceDirection::REVERSE;

        if (X.get_partial_shape().is_dynamic() || H_t.get_partial_shape().is_dynamic()
            || seq_lengths.get_partial_shape().is_dynamic()) {
            return false;
        }

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        auto max_seq_len = X.get_shape().at(1);
        bool enable_mask = should_enable_mask(seq_lengths, max_seq_len);

        std::shared_ptr<Node> reverse_seq;
        if (is_reverse && enable_mask) {
            reverse_seq = std::make_shared<opset5::ReverseSequence>(X, seq_lengths, 0, 1);
        }
        // TensorIterator Body: begin
        Shape X_param_shape = X.get_shape();
        X_param_shape.at(1) = 1; // split by seq_lengths dimension
        auto X_body_param = std::make_shared<opset5::Parameter>(X.get_element_type(), X_param_shape);
        auto H_body_param = std::make_shared<opset5::Parameter>(H_t.get_element_type(),
                                                                H_t.get_shape());
        auto seq_body_param = std::make_shared<opset5::Parameter>(seq_lengths.get_element_type(),
                                                                  seq_lengths.get_partial_shape());

        auto axis_0 = std::make_shared<op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
        auto axis_1 = std::make_shared<op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});

        const auto& ins = squeeze_nodes({X_body_param, H_body_param, W, R, B}, {axis_1, axis_1, axis_0, axis_0, axis_0});
        auto cell = std::make_shared<opset5::RNNCell>(ins[0],
                                                      ins[1],
                                                      ins[2],
                                                      ins[3],
                                                      ins[4],
                                                      rnn_seq->get_hidden_size(),
                                                      rnn_seq->get_activations(),
                                                      rnn_seq->get_activations_alpha(),
                                                      rnn_seq->get_activations_beta(),
                                                      rnn_seq->get_clip());

        ParameterVector body_params;
        ResultVector body_results;
        auto unsqueeze_dum_dir = std::make_shared<opset5::Unsqueeze>(cell->output(0), axis_1);
        Output<Node> h_node_to_result = unsqueeze_dum_dir;
        if (enable_mask) {
            auto current_iter = get_current_iter(body_params, body_results, seq_body_param);
            h_node_to_result = get_masked_value(tensor_iterator, body_params, body_results, current_iter,
                                                unsqueeze_dum_dir, seq_body_param);
        }

        auto H_res = std::make_shared<opset5::Result>(h_node_to_result);
        auto unsqueeze_seq_len = std::make_shared<opset5::Unsqueeze>(h_node_to_result, axis_1);
        auto concat_res = std::make_shared<opset5::Result>(unsqueeze_seq_len);
        body_params.push_back(X_body_param);
        body_params.push_back(H_body_param);
        body_params.push_back(seq_body_param);
        body_results.push_back(concat_res);
        body_results.push_back(H_res);

        auto body = std::make_shared<ngraph::Function>(body_results, body_params);
        tensor_iterator->set_function(body);
        // TensorIterator Body: end
        if (is_reverse) {
            if (!enable_mask) {
                tensor_iterator->set_sliced_input(X_body_param, X, -1, -1, 1, 0, 1);
                tensor_iterator->get_concatenated_slices(concat_res, -1, -1, 1, 0, 2);
            } else {
                tensor_iterator->set_sliced_input(X_body_param, reverse_seq, 0, 1, 1, -1, 1);
                tensor_iterator->get_concatenated_slices(concat_res, 0, 1, 1, -1, 2);
            }
        } else {
            tensor_iterator->set_sliced_input(X_body_param, X, 0, 1, 1, -1, 1);
            tensor_iterator->get_concatenated_slices(concat_res, 0, 1, 1, -1, 2);
        }
        tensor_iterator->set_merged_input(H_body_param, H_t, H_res);
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);
        Output<Node> H_out = H_res;
        if (enable_mask) {
            // create initial values for body_parameters in outer graph
            // aggregated Y_h - concatenation of the last non-zero values for each batch
            auto aggregated_Y_h = std::make_shared<ngraph::opset5::Constant>(H_body_param->get_element_type(),
                                                                             H_body_param->get_shape(),
                                                                             std::vector<float>(shape_size(H_body_param->get_shape()),
                                                                                                0.f));
            auto init_val_curr_iter = std::make_shared<ngraph::opset5::Constant>(seq_lengths.get_element_type(),
                                                                                 ngraph::Shape{1},
                                                                                 std::vector<int64_t>{1});

            // set initial value and back edge for current iteration
            tensor_iterator->set_merged_input(body_params.at(0), init_val_curr_iter, body_results.at(0));
            // set initial value and back edge for aggregated H
            tensor_iterator->set_merged_input(body_params.at(1), aggregated_Y_h, body_results.at(1));
            H_out = tensor_iterator->get_function()->get_results()[1];
        }
        tensor_iterator->get_iter_value(H_out);
        tensor_iterator->set_friendly_name(rnn_seq->get_friendly_name());
        if (enable_mask && is_reverse) {
            auto reverse_seq = std::make_shared<opset5::ReverseSequence>(tensor_iterator->output(0), seq_lengths, 0, 2);
            auto identity_1 = std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), axis_1);
            auto identity_2 = std::make_shared<opset5::Squeeze>(identity_1, axis_1);
            ngraph::copy_runtime_info(rnn_seq, {reverse_seq, tensor_iterator});
            ngraph::replace_node(rnn_seq, {reverse_seq, identity_2});
            tensor_iterator->set_friendly_name(rnn_seq->get_friendly_name() + "/tensor_iterator");
            reverse_seq->set_friendly_name(rnn_seq->get_friendly_name() + ".0");
            identity_2->set_friendly_name(rnn_seq->get_friendly_name() + ".1");
        } else {
            ngraph::copy_runtime_info(rnn_seq, tensor_iterator);
            ngraph::replace_node(rnn_seq, tensor_iterator);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_seq, "ConvertRNNSequenceToTensorIterator");
    register_matcher(m, callback);
}

ngraph::pass::ConvertGRUSequenceToTensorIterator::ConvertGRUSequenceToTensorIterator() {
    auto rnn_seq = ngraph::pattern::wrap_type<opset5::GRUSequence>();
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto rnn_seq = std::dynamic_pointer_cast<ngraph::opset5::GRUSequence>(m.get_match_root());

        // Bidirectional Sequence op should be decomposed to Reverse + Forward
        // (e.g. apply BidirectionalRNNSequenceDecomposition transformation before this one)
        if (!rnn_seq && rnn_seq->get_direction() != ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
            return false;
        }

        NodeVector new_nodes;
        const auto &X = rnn_seq->input_value(0); // split
        const auto &H_t = rnn_seq->input_value(1); // merged (init value + back edge)
        const auto &seq_lengths = rnn_seq->input_value(2); // invariant
        const auto &W = rnn_seq->input_value(3); // const in the body
        const auto &R = rnn_seq->input_value(4); // const in the body
        const auto &B = rnn_seq->input_value(5); // const in the body
        bool is_reverse = rnn_seq->get_direction() == ngraph::op::RecurrentSequenceDirection::REVERSE;

        if (X.get_partial_shape().is_dynamic() || H_t.get_partial_shape().is_dynamic()
            || seq_lengths.get_partial_shape().is_dynamic()) {
            return false;
        }

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        auto max_seq_len = X.get_shape().at(1);
        bool enable_mask = should_enable_mask(seq_lengths, max_seq_len);

        std::shared_ptr<Node> reverse_seq;
        if (is_reverse && enable_mask) {
            reverse_seq = std::make_shared<opset5::ReverseSequence>(X, seq_lengths, 0, 1);
        }
        // TensorIterator Body: begin
        Shape X_param_shape = X.get_shape();
        X_param_shape.at(1) = 1; // split by seq_lengths dimension
        auto X_body_param = std::make_shared<opset5::Parameter>(X.get_element_type(), X_param_shape);
        auto H_body_param = std::make_shared<opset5::Parameter>(H_t.get_element_type(),
                                                                H_t.get_shape());
        auto seq_body_param = std::make_shared<opset5::Parameter>(seq_lengths.get_element_type(),
                                                                  seq_lengths.get_partial_shape());

        auto axis_0 = std::make_shared<op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
        auto axis_1 = std::make_shared<op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});

        const auto& ins = squeeze_nodes({X_body_param, H_body_param, W, R, B}, {axis_1, axis_1, axis_0, axis_0, axis_0});
        auto cell = std::make_shared<opset5::GRUCell>(ins[0],
                                                      ins[1],
                                                      ins[2],
                                                      ins[3],
                                                      ins[4],
                                                      rnn_seq->get_hidden_size(),
                                                      rnn_seq->get_activations(),
                                                      rnn_seq->get_activations_alpha(),
                                                      rnn_seq->get_activations_beta(),
                                                      rnn_seq->get_clip(),
                                                      rnn_seq->get_linear_before_reset());

        ParameterVector body_params;
        ResultVector body_results;
        auto unsqueeze_dum_dir = std::make_shared<opset5::Unsqueeze>(cell->output(0), axis_1);
        Output<Node> h_node_to_result = unsqueeze_dum_dir;
        if (enable_mask) {
            auto current_iter = get_current_iter(body_params, body_results, seq_body_param);
            h_node_to_result = get_masked_value(tensor_iterator, body_params, body_results, current_iter,
                                                unsqueeze_dum_dir, seq_body_param);
        }

        auto H_res = std::make_shared<opset5::Result>(h_node_to_result);
        auto unsqueeze_seq_len = std::make_shared<opset5::Unsqueeze>(h_node_to_result, axis_1);
        auto concat_res = std::make_shared<opset5::Result>(unsqueeze_seq_len);
        body_params.push_back(X_body_param);
        body_params.push_back(H_body_param);
        body_params.push_back(seq_body_param);
        body_results.push_back(concat_res);
        body_results.push_back(H_res);

        auto body = std::make_shared<ngraph::Function>(body_results, body_params);
        tensor_iterator->set_function(body);
        // TensorIterator Body: end
        if (is_reverse) {
            if (!enable_mask) {
                tensor_iterator->set_sliced_input(X_body_param, X, -1, -1, 1, 0, 1);
                tensor_iterator->get_concatenated_slices(concat_res, -1, -1, 1, 0, 2);
            } else {
                tensor_iterator->set_sliced_input(X_body_param, reverse_seq, 0, 1, 1, -1, 1);
                tensor_iterator->get_concatenated_slices(concat_res, 0, 1, 1, -1, 2);
            }
        } else {
            tensor_iterator->set_sliced_input(X_body_param, X, 0, 1, 1, -1, 1);
            tensor_iterator->get_concatenated_slices(concat_res, 0, 1, 1, -1, 2);
        }
        tensor_iterator->set_merged_input(H_body_param, H_t, H_res);
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);
        Output<Node> H_out = H_res;
        if (enable_mask) {
            // create initial values for body_parameters in outer graph
            // aggregated Y_h - concatenation of the last non-zero values for each batch
            auto aggregated_Y_h = std::make_shared<ngraph::opset5::Constant>(H_body_param->get_element_type(),
                                                                             H_body_param->get_shape(),
                                                                             std::vector<float>(shape_size(H_body_param->get_shape()),
                                                                                                0.f));
            auto init_val_curr_iter = std::make_shared<ngraph::opset5::Constant>(seq_lengths.get_element_type(),
                                                                                 ngraph::Shape{1},
                                                                                 std::vector<int64_t>{1});

            // set initial value and back edge for current iteration
            tensor_iterator->set_merged_input(body_params.at(0), init_val_curr_iter, body_results.at(0));
            // set initial value and back edge for aggregated H
            tensor_iterator->set_merged_input(body_params.at(1), aggregated_Y_h, body_results.at(1));
            H_out = tensor_iterator->get_function()->get_results()[1];
        }
        tensor_iterator->get_iter_value(H_out);
        tensor_iterator->set_friendly_name(rnn_seq->get_friendly_name());
        if (enable_mask && is_reverse) {
            auto reverse_seq = std::make_shared<opset5::ReverseSequence>(tensor_iterator->output(0), seq_lengths, 0, 2);
            auto identity_1 = std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), axis_1);
            auto identity_2 = std::make_shared<opset5::Squeeze>(identity_1, axis_1);
            ngraph::copy_runtime_info(rnn_seq, {reverse_seq, tensor_iterator});
            ngraph::replace_node(rnn_seq, {reverse_seq, identity_2});
            tensor_iterator->set_friendly_name(rnn_seq->get_friendly_name() + "/tensor_iterator");
            reverse_seq->set_friendly_name(rnn_seq->get_friendly_name() + ".0");
            identity_2->set_friendly_name(rnn_seq->get_friendly_name() + ".1");
        } else {
            ngraph::copy_runtime_info(rnn_seq, tensor_iterator);
            ngraph::replace_node(rnn_seq, tensor_iterator);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_seq, "ConvertGRUSequenceToTensorIterator");
    register_matcher(m, callback);
}

ngraph::pass::ConvertLSTMSequenceToTensorIterator::ConvertLSTMSequenceToTensorIterator() {
    auto rnn_seq = ngraph::pattern::wrap_type<opset5::LSTMSequence>();
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto rnn_seq = std::dynamic_pointer_cast<ngraph::opset5::LSTMSequence>(m.get_match_root());

        // Bidirectional Sequence op should be decomposed to Reverse + Forward
        // (e.g. apply BidirectionalRNNSequenceDecomposition transformation before this one)
        if (!rnn_seq && rnn_seq->get_direction() != ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
            return false;
        }

        NodeVector new_nodes;
        const auto &X = rnn_seq->input_value(0); // split
        const auto &H_t = rnn_seq->input_value(1); // merged (init value + back edge)
        const auto &C_t = rnn_seq->input_value(2); // merged (init value + back edge)
        const auto &seq_lengths = rnn_seq->input_value(3); // invariant
        const auto &W = rnn_seq->input_value(4); // const in the body
        const auto &R = rnn_seq->input_value(5); // const in the body
        const auto &B = rnn_seq->input_value(6); // const in the body
        bool is_reverse = rnn_seq->get_direction() == ngraph::op::RecurrentSequenceDirection::REVERSE;

        if (X.get_partial_shape().is_dynamic() || H_t.get_partial_shape().is_dynamic()
            || seq_lengths.get_partial_shape().is_dynamic()) {
            return false;
        }

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        auto max_seq_len = X.get_shape().at(1);
        bool enable_mask = should_enable_mask(seq_lengths, max_seq_len);

        std::shared_ptr<Node> reverse_seq;
        if (is_reverse && enable_mask) {
            reverse_seq = std::make_shared<opset5::ReverseSequence>(X, seq_lengths, 0, 1);
        }
        // TensorIterator Body: begin
        Shape X_param_shape = X.get_shape();
        X_param_shape.at(1) = 1; // split by seq_lengths dimension
        auto X_body_param = std::make_shared<opset5::Parameter>(X.get_element_type(), X_param_shape);
        auto H_body_param = std::make_shared<opset5::Parameter>(H_t.get_element_type(),
                                                                H_t.get_shape());
        auto C_body_param = std::make_shared<opset5::Parameter>(C_t.get_element_type(),
                                                                C_t.get_shape());
        auto seq_body_param = std::make_shared<opset5::Parameter>(seq_lengths.get_element_type(),
                                                                  seq_lengths.get_partial_shape());

        auto axis_0 = std::make_shared<op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
        auto axis_1 = std::make_shared<op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});

        const auto& ins = squeeze_nodes({X_body_param, H_body_param, C_body_param, W, R, B},
                                        {axis_1, axis_1, axis_1, axis_0, axis_0, axis_0});
        auto cell = std::make_shared<opset5::LSTMCell>(ins[0],
                                                      ins[1],
                                                      ins[2],
                                                      ins[3],
                                                      ins[4],
                                                      ins[5],
                                                      rnn_seq->get_hidden_size(),
                                                      rnn_seq->get_activations(),
                                                      rnn_seq->get_activations_alpha(),
                                                      rnn_seq->get_activations_beta(),
                                                      rnn_seq->get_clip());

        ParameterVector body_params;
        ResultVector body_results;
        auto unsqueeze_dum_dir_h = std::make_shared<opset5::Unsqueeze>(cell->output(0), axis_1);
        auto unsqueeze_dum_dir_c = std::make_shared<opset5::Unsqueeze>(cell->output(1), axis_1);
        Output<Node> h_node_to_result = unsqueeze_dum_dir_h;
        Output<Node> c_node_to_result = unsqueeze_dum_dir_c;
        if (enable_mask) {
            auto current_iter = get_current_iter(body_params, body_results, seq_body_param);
            h_node_to_result = get_masked_value(tensor_iterator, body_params, body_results, current_iter,
                                                unsqueeze_dum_dir_h, seq_body_param);
            c_node_to_result = get_masked_value(tensor_iterator, body_params, body_results, current_iter,
                                                unsqueeze_dum_dir_c, seq_body_param);
        }

        auto H_res = std::make_shared<opset5::Result>(h_node_to_result);
        auto C_res = std::make_shared<opset5::Result>(c_node_to_result);
        auto unsqueeze_seq_len = std::make_shared<opset5::Unsqueeze>(h_node_to_result, axis_1);
        auto concat_res = std::make_shared<opset5::Result>(unsqueeze_seq_len);
        body_params.push_back(X_body_param);
        body_params.push_back(H_body_param);
        body_params.push_back(C_body_param);
        body_params.push_back(seq_body_param);
        body_results.push_back(concat_res);
        body_results.push_back(H_res);
        body_results.push_back(C_res);

        auto body = std::make_shared<ngraph::Function>(body_results, body_params);
        tensor_iterator->set_function(body);
        // TensorIterator Body: end
        if (is_reverse) {
            if (!enable_mask) {
                tensor_iterator->set_sliced_input(X_body_param, X, -1, -1, 1, 0, 1);
                tensor_iterator->get_concatenated_slices(concat_res, -1, -1, 1, 0, 2);
            } else {
                tensor_iterator->set_sliced_input(X_body_param, reverse_seq, 0, 1, 1, -1, 1);
                tensor_iterator->get_concatenated_slices(concat_res, 0, 1, 1, -1, 2);
            }
        } else {
            tensor_iterator->set_sliced_input(X_body_param, X, 0, 1, 1, -1, 1);
            tensor_iterator->get_concatenated_slices(concat_res, 0, 1, 1, -1, 2);
        }
        tensor_iterator->set_merged_input(H_body_param, H_t, H_res);
        tensor_iterator->set_merged_input(C_body_param, C_t, C_res);
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);
        Output<Node> H_out = H_res;
        Output<Node> C_out = C_res;
        if (enable_mask) {
            // create initial values for body_parameters in outer graph
            // aggregated Y_h - concatenation of the last non-zero values for each batch
            auto aggregated_Y_h = std::make_shared<ngraph::opset5::Constant>(H_body_param->get_element_type(),
                                                                             H_body_param->get_shape(),
                                                                             std::vector<float>(shape_size(H_body_param->get_shape()),
                                                                                                0.f));
            auto aggregated_Y_c = std::make_shared<ngraph::opset5::Constant>(C_body_param->get_element_type(),
                                                                             C_body_param->get_shape(),
                                                                             std::vector<float>(shape_size(C_body_param->get_shape()),
                                                                                                0.f));
            auto init_val_curr_iter = std::make_shared<ngraph::opset5::Constant>(seq_lengths.get_element_type(),
                                                                                 ngraph::Shape{1},
                                                                                 std::vector<int64_t>{1});

            // set initial value and back edge for current iteration
            tensor_iterator->set_merged_input(body_params.at(0), init_val_curr_iter, body_results.at(0));
            // set initial value and back edge for aggregated H
            tensor_iterator->set_merged_input(body_params.at(1), aggregated_Y_h, body_results.at(1));
            // set initial value and back edge for aggregated H
            tensor_iterator->set_merged_input(body_params.at(2), aggregated_Y_c, body_results.at(2));
            H_out = tensor_iterator->get_function()->get_results()[1];
            C_out = tensor_iterator->get_function()->get_results()[2];
        }
        tensor_iterator->get_iter_value(H_out);
        tensor_iterator->get_iter_value(C_out);
        tensor_iterator->set_friendly_name(rnn_seq->get_friendly_name());
        if (enable_mask && is_reverse) {
            auto reverse_seq = std::make_shared<opset5::ReverseSequence>(tensor_iterator->output(0), seq_lengths, 0, 2);
            auto identity_1_h = std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), axis_1);
            auto identity_2_h = std::make_shared<opset5::Squeeze>(identity_1_h, axis_1);

            auto identity_1_c = std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(2), axis_1);
            auto identity_2_c = std::make_shared<opset5::Squeeze>(identity_1_c, axis_1);

            ngraph::copy_runtime_info(rnn_seq, {reverse_seq, tensor_iterator});
            ngraph::replace_node(rnn_seq, {reverse_seq, identity_2_h, identity_2_c});
            tensor_iterator->set_friendly_name(rnn_seq->get_friendly_name() + "/tensor_iterator");
            reverse_seq->set_friendly_name(rnn_seq->get_friendly_name() + ".0");
            identity_2_h->set_friendly_name(rnn_seq->get_friendly_name() + ".1");
            identity_2_c->set_friendly_name(rnn_seq->get_friendly_name() + ".2");
        } else {
            ngraph::copy_runtime_info(rnn_seq, tensor_iterator);
            ngraph::replace_node(rnn_seq, tensor_iterator);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_seq, "ConvertGRUSequenceToTensorIterator");
    register_matcher(m, callback);
}