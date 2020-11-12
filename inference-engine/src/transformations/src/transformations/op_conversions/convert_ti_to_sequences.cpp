// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/specialize_function.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertTensorIteratorToLSTMSequence, "ConvertTensorIteratorToLSTMSequence", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertTensorIteratorToRNNSequence, "ConvertTensorIteratorToRNNSequence", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertTensorIteratorToGRUSequence, "ConvertTensorIteratorToGRUSequence", 0);

ngraph::pass::ConvertTensorIteratorToLSTMSequence::ConvertTensorIteratorToLSTMSequence() {
    auto tensor_iterator = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32,
                                                                        ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset5::TensorIterator>());
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher &m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator>(m.get_match_root());
        if (!ti || !m_transformation_callback(ti))
            return false;

        // create pattern
        auto data = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1});
        auto axis_squeeze = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 1);

        auto input_data = std::make_shared<ngraph::opset5::Squeeze>(data, axis_squeeze);
        auto input_H_state = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_C_state = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_W = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{4, 1});
        auto input_R = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{4, 1});
        auto input_B = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{4});

        auto cell = std::make_shared<ngraph::opset5::LSTMCell>(input_data, input_H_state, input_C_state,
                                                               input_W, input_R, input_B, 1);

        auto axis_unsqueeze = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 1);
        auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(cell, axis_unsqueeze);
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

        auto pattern_map = matcher.get_pattern_map();

        auto params = func->get_parameters();
        std::vector<std::shared_ptr<ngraph::opset5::TensorIterator::InputDescription>> ordered_in_descs(3);
        int64_t stride = 0, slice_axis = 0;
        size_t batch_size = 0;
        for (const auto& input_desc : ti->get_input_descriptions()) {
            auto param = params[input_desc->m_body_parameter_index];
            if (param == pattern_map[data]) {
                // to get batch size value
                if (param->get_partial_shape().is_dynamic()) {
                    return false;
                }
                auto slice_input
                        = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::SliceInputDescription>(input_desc);
                if (!slice_input)
                    return false;

                stride = slice_input->m_stride;
                slice_axis = slice_input->m_axis;

                if (!(slice_axis == 0 || slice_axis == 1)) {
                    return false;
                }
                batch_size = param->get_shape()[slice_axis == 0 ? 1 : 0];
                ordered_in_descs[0] = input_desc;
            } else if (param == pattern_map[input_H_state]) {
                ordered_in_descs[1] = input_desc;
            } else if (param == pattern_map[input_C_state]) {
                ordered_in_descs[2] = input_desc;
            } else {
                return false;
            }
        }

        auto results = func->get_results();
        std::vector<std::shared_ptr<ngraph::opset5::TensorIterator::OutputDescription>> ordered_out_descs(3);
        for (const auto& output_desc : ti->get_output_descriptions()) {
            std::shared_ptr<opset5::Result> res = results[output_desc->m_body_value_index];
            if (res->get_input_source_output(0) == pattern_map[unsqueeze]) {
                auto concat_output
                        = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::ConcatOutputDescription>(output_desc);
                if (!concat_output)
                    return false;

                stride = concat_output->m_stride;
                ordered_out_descs[0] = output_desc;
            } else if (res->get_input_source_output(0) == pattern_map[cell]->output(0)) {
                ordered_out_descs[1] = output_desc;
            } else if (res->get_input_source_output(0) == pattern_map[cell]->output(1)) {
                ordered_out_descs[2] = output_desc;
            } else {
                return false;
            }
        }

        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{batch_size}, {ti->get_num_iterations()});
        const auto& lstm_cell = std::dynamic_pointer_cast<ngraph::opset5::LSTMCell>(pattern_map[cell]);
        auto in_0 = ti->input_values()[ordered_in_descs[0]->m_input_index];
        if (slice_axis == 0) {
            auto order = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 0, 2});
            in_0 = std::make_shared<ngraph::opset5::Transpose>(ti->input_values()[ordered_in_descs[0]->m_input_index], order);
        }
        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset5::Unsqueeze>(ti->input_values()[ordered_in_descs[1]->m_input_index], axis_1);
        auto in_2 = std::make_shared<ngraph::opset5::Unsqueeze>(ti->input_values()[ordered_in_descs[2]->m_input_index], axis_1);

        auto axis_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_4 = std::make_shared<ngraph::opset5::Unsqueeze>(pattern_map[input_W]->output(0).get_node_shared_ptr(), axis_2);
        auto in_5 = std::make_shared<ngraph::opset5::Unsqueeze>(pattern_map[input_R]->output(0).get_node_shared_ptr(), axis_2);
        auto in_6 = std::make_shared<ngraph::opset5::Unsqueeze>(pattern_map[input_B]->output(0).get_node_shared_ptr(), axis_2);
        auto sequence = std::make_shared<opset5::LSTMSequence>(
                in_0,
                in_1,
                in_2,
                seq_lengths,
                in_4,
                in_5,
                in_6,
                lstm_cell->get_hidden_size(),
                stride > 0 ? ngraph::op::RecurrentSequenceDirection::FORWARD: ngraph::op::RecurrentSequenceDirection::REVERSE,
                lstm_cell->get_activations_alpha(),
                lstm_cell->get_activations_beta(),
                lstm_cell->get_activations(),
                lstm_cell->get_clip());

        auto axis_out = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto out_0 = std::make_shared<ngraph::opset5::Squeeze>(sequence->output(0), axis_out);
        auto out_1 = std::make_shared<ngraph::opset5::Squeeze>(sequence->output(1), axis_out);
        auto out_2 = std::make_shared<ngraph::opset5::Squeeze>(sequence->output(2), axis_out);

        std::shared_ptr<Node> out = out_0;
        if (slice_axis == 0) {
            auto order = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 0, 2});
            out = std::make_shared<ngraph::opset5::Transpose>(out_0, order);
        }

        ngraph::NodeVector outputs = {out, out_1, out_2};
        for (size_t i = 0; i < ordered_out_descs.size(); ++i) {
            if (ordered_out_descs[i]) {
                for (const auto &input : ti->output(ordered_out_descs[i]->m_output_index).get_target_inputs()) {
                    input.replace_source_output(outputs[i]->output(0));
                }
                outputs[i]->get_output_tensor(0).set_name(op::util::create_ie_output_name(ti->output(ordered_out_descs[i]->m_output_index)));
            }
        }

        ngraph::NodeVector new_nodes = {in_1, in_2, in_4, in_5, in_6, sequence, out_0, out_1, out_2};
        if (slice_axis == 0) {
            new_nodes.push_back(out);
            new_nodes.push_back(in_0.get_node_shared_ptr());
        }
        copy_runtime_info(ti, new_nodes);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ConvertTensorIteratorToLSTMSequence");
    register_matcher(m, callback);
}

ngraph::pass::ConvertTensorIteratorToRNNSequence::ConvertTensorIteratorToRNNSequence() {
    auto tensor_iterator = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32,
                                                                        ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset5::TensorIterator>());
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher &m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator>(m.get_match_root());
        if (!ti || !m_transformation_callback(ti))
            return false;

        // create pattern
        auto data = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1});
        auto axis_squeeze = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
        auto input_data = std::make_shared<ngraph::opset5::Squeeze>(data, axis_squeeze);

        auto input_H_state = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_W = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_R = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_B = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{1});

        auto cell = std::make_shared<ngraph::opset5::RNNCell>(input_data, input_H_state, input_W, input_R, input_B, 1);

        auto axis_unsqueeze = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
        auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(cell, axis_unsqueeze);
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

        auto pattern_map = matcher.get_pattern_map();

        auto params = func->get_parameters();
        std::vector<std::shared_ptr<ngraph::opset5::TensorIterator::InputDescription>> ordered_in_descs(3);
        int64_t stride = 0, slice_axis = 0;
        size_t batch_size = 0;
        for (const auto& input_desc : ti->get_input_descriptions()) {
            auto param = params[input_desc->m_body_parameter_index];
            if (param == pattern_map[data]) {
                // to get batch size value
                if (param->get_partial_shape().is_dynamic()) {
                    return false;
                }
                auto slice_input
                        = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::SliceInputDescription>(input_desc);
                if (!slice_input)
                    return false;

                stride = slice_input->m_stride;
                slice_axis = slice_input->m_axis;
                if (!(slice_axis == 0 || slice_axis == 1)) {
                    return false;
                }
                batch_size = param->get_shape()[slice_axis == 0 ? 1 : 0];
                ordered_in_descs[0] = input_desc;
            } else if (param == pattern_map[input_H_state]) {
                ordered_in_descs[1] = input_desc;
            } else {
                return false;
            }
        }

        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{batch_size}, {ti->get_num_iterations()});

        auto results = func->get_results();
        std::vector<std::shared_ptr<ngraph::opset5::TensorIterator::OutputDescription>> ordered_out_descs(2);
        for (const auto& output_desc : ti->get_output_descriptions()) {
            std::shared_ptr<opset5::Result> res = results[output_desc->m_body_value_index];
            if (res->get_input_source_output(0) == pattern_map[unsqueeze]) {
                auto concat_output
                        = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::ConcatOutputDescription>(output_desc);
                if (!concat_output)
                    return false;

                stride = concat_output->m_stride;
                ordered_out_descs[0] = output_desc;
            } else if (res->get_input_source_output(0) == pattern_map[cell]->output(0)) {
                ordered_out_descs[1] = output_desc;
            } else {
                return false;
            }
        }

        const auto& rnn_cell = std::dynamic_pointer_cast<ngraph::opset5::RNNCell>(pattern_map[cell]);

        auto in_0 = ti->input_values()[ordered_in_descs[0]->m_input_index];
        if (slice_axis == 0) {
            auto order = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 0, 2});
            in_0 = std::make_shared<ngraph::opset5::Transpose>(ti->input_values()[ordered_in_descs[0]->m_input_index], order);
        }

        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset5::Unsqueeze>(ti->input_values()[ordered_in_descs[1]->m_input_index], axis_1);

        auto axis_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset5::Unsqueeze>(pattern_map[input_W]->output(0).get_node_shared_ptr(), axis_2);
        auto in_4 = std::make_shared<ngraph::opset5::Unsqueeze>(pattern_map[input_R]->output(0).get_node_shared_ptr(), axis_2);
        auto in_5 = std::make_shared<ngraph::opset5::Unsqueeze>(pattern_map[input_B]->output(0).get_node_shared_ptr(), axis_2);
        auto sequence = std::make_shared<opset5::RNNSequence>(
                in_0,
                in_1,
                seq_lengths,
                in_3,
                in_4,
                in_5,
                rnn_cell->get_hidden_size(),
                stride > 0 ? ngraph::op::RecurrentSequenceDirection::FORWARD: ngraph::op::RecurrentSequenceDirection::REVERSE,
                rnn_cell->get_activations(),
                rnn_cell->get_activations_alpha(),
                rnn_cell->get_activations_beta(),
                rnn_cell->get_clip());

        auto axis_out = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto out_0 = std::make_shared<ngraph::opset5::Squeeze>(sequence->output(0), axis_out);
        auto out_1 = std::make_shared<ngraph::opset5::Squeeze>(sequence->output(1), axis_out);

        std::shared_ptr<Node> out = out_0;
        if (slice_axis == 0) {
            auto order = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 0, 2});
            out = std::make_shared<ngraph::opset5::Transpose>(out_0, order);
        }

        ngraph::NodeVector outputs = {out, out_1};
        for (size_t i = 0; i < ordered_out_descs.size(); ++i) {
            if (ordered_out_descs[i]) {
                for (const auto &input : ti->output(ordered_out_descs[i]->m_output_index).get_target_inputs()) {
                    input.replace_source_output(outputs[i]->output(0));
                }
                outputs[i]->get_output_tensor(0).set_name(op::util::create_ie_output_name(ti->output(ordered_out_descs[i]->m_output_index)));
            }
        }

        ngraph::OutputVector new_nodes = {in_1, in_3, in_4, in_5, sequence, out_0, out_1};
        if (slice_axis == 0) {
            new_nodes.push_back(out);
            new_nodes.push_back(in_0);
        }
        copy_runtime_info(ti, as_node_vector(new_nodes));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ConvertTensorIteratorToRNNSequence");
    register_matcher(m, callback);
}

ngraph::pass::ConvertTensorIteratorToGRUSequence::ConvertTensorIteratorToGRUSequence() {
    auto tensor_iterator = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32,
                                                                        ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset5::TensorIterator>());
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher &m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator>(m.get_match_root());
        if (!ti || !m_transformation_callback(ti))
            return false;

        // create pattern
        auto data = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1});
        auto axis_squeeze = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
        auto input_data = std::make_shared<ngraph::opset5::Squeeze>(data, axis_squeeze);

        auto input_H_state = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_W = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{3, 1});
        auto input_R = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{3, 1});
        auto input_B = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{3});

        auto cell = std::make_shared<ngraph::opset5::GRUCell>(input_data, input_H_state, input_W, input_R, input_B, 1);

        auto axis_unsqueeze = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
        auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(cell, axis_unsqueeze);
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

        auto pattern_map = matcher.get_pattern_map();

        auto params = func->get_parameters();
        std::vector<std::shared_ptr<ngraph::opset5::TensorIterator::InputDescription>> ordered_in_descs(3);
        int64_t stride = 0, slice_axis = 0;
        size_t batch_size = 0;
        for (const auto& input_desc : ti->get_input_descriptions()) {
            auto param = params[input_desc->m_body_parameter_index];
            if (param == pattern_map[data]) {
                // to get batch size value
                if (param->get_partial_shape().is_dynamic()) {
                    return false;
                }
                auto slice_input
                        = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::SliceInputDescription>(input_desc);
                if (!slice_input)
                    return false;

                stride = slice_input->m_stride;
                slice_axis = slice_input->m_axis;
                if (!(slice_axis == 0 || slice_axis == 1)) {
                    return false;
                }
                batch_size = param->get_shape()[slice_axis == 0 ? 1 : 0];
                ordered_in_descs[0] = input_desc;
            } else if (param == pattern_map[input_H_state]) {
                ordered_in_descs[1] = input_desc;
            } else {
                return false;
            }
        }

        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{batch_size}, {ti->get_num_iterations()});

        auto results = func->get_results();
        std::vector<std::shared_ptr<ngraph::opset5::TensorIterator::OutputDescription>> ordered_out_descs(2);
        for (const auto& output_desc : ti->get_output_descriptions()) {
            std::shared_ptr<opset5::Result> res = results[output_desc->m_body_value_index];
            if (res->get_input_source_output(0) == pattern_map[unsqueeze]) {
                auto concat_output
                        = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::ConcatOutputDescription>(output_desc);
                if (!concat_output)
                    return false;

                stride = concat_output->m_stride;
                ordered_out_descs[0] = output_desc;
            } else if (res->get_input_source_output(0) == pattern_map[cell]->output(0)) {
                ordered_out_descs[1] = output_desc;
            } else {
                return false;
            }
        }

        const auto& rnn_cell = std::dynamic_pointer_cast<ngraph::opset5::GRUCell>(pattern_map[cell]);

        auto in_0 = ti->input_values()[ordered_in_descs[0]->m_input_index];
        if (slice_axis == 0) {
            auto order = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 0, 2});
            in_0 = std::make_shared<ngraph::opset5::Transpose>(ti->input_values()[ordered_in_descs[0]->m_input_index], order);
        }

        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset5::Unsqueeze>(ti->input_values()[ordered_in_descs[1]->m_input_index], axis_1);

        auto axis_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset5::Unsqueeze>(pattern_map[input_W]->output(0).get_node_shared_ptr(), axis_2);
        auto in_4 = std::make_shared<ngraph::opset5::Unsqueeze>(pattern_map[input_R]->output(0).get_node_shared_ptr(), axis_2);
        auto in_5 = std::make_shared<ngraph::opset5::Unsqueeze>(pattern_map[input_B]->output(0).get_node_shared_ptr(), axis_2);
        auto sequence = std::make_shared<opset5::GRUSequence>(
                in_0,
                in_1,
                seq_lengths,
                in_3,
                in_4,
                in_5,
                rnn_cell->get_hidden_size(),
                stride > 0 ? ngraph::op::RecurrentSequenceDirection::FORWARD: ngraph::op::RecurrentSequenceDirection::REVERSE,
                rnn_cell->get_activations(),
                rnn_cell->get_activations_alpha(),
                rnn_cell->get_activations_beta(),
                rnn_cell->get_clip(),
                rnn_cell->get_linear_before_reset());

        auto axis_out = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto out_0 = std::make_shared<ngraph::opset5::Squeeze>(sequence->output(0), axis_out);
        auto out_1 = std::make_shared<ngraph::opset5::Squeeze>(sequence->output(1), axis_out);

        std::shared_ptr<Node> out = out_0;
        if (slice_axis == 0) {
            auto order = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 0, 2});
            out = std::make_shared<ngraph::opset5::Transpose>(out_0, order);
        }

        ngraph::NodeVector outputs = {out, out_1};
        for (size_t i = 0; i < ordered_out_descs.size(); ++i) {
            if (ordered_out_descs[i]) {
                for (const auto &input : ti->output(ordered_out_descs[i]->m_output_index).get_target_inputs()) {
                    input.replace_source_output(outputs[i]->output(0));
                }
                outputs[i]->get_output_tensor(0).set_name(op::util::create_ie_output_name(ti->output(ordered_out_descs[i]->m_output_index)));
            }
        }

        ngraph::OutputVector new_nodes = {in_1, in_3, in_4, in_5, sequence, out_0, out_1};
        if (slice_axis == 0) {
            new_nodes.push_back(out);
            new_nodes.push_back(in_0);
        }
        copy_runtime_info(ti, as_node_vector(new_nodes));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ConvertTensorIteratorToGRUSequence");
    register_matcher(m, callback);
}