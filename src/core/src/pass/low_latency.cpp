// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/low_latency.hpp"

#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"

namespace {
std::string generate_variable_name(const std::string& op_name, const std::string& param_name, int64_t variable_idx) {
    return op_name + "/" + param_name + "/" + "variable_" + std::to_string(variable_idx);
}

}  // namespace

namespace {

const std::string msg_low_latency_2_already_applied = "LowLatency2 transformation cannot be applied because the "
                                                      "ReadValue node is already an input to the TensorIterator."
                                                      "LowLatency2 transformation may have already been applied, please"
                                                      "do not call it more then once.";
const std::string msg_low_latency_already_applied = "LowLatency2 transformation cannot be applied because the "
                                                    "ReadValue node is already inside the TensorIterator. "
                                                    "LowLatency transformation may have been applied, please do "
                                                    "not call LowLatency2 after LowLatency.";

void unroll_single_iteration(const std::shared_ptr<ov::op::util::SubGraphOp>& sub_graph_op,
                             const std::shared_ptr<ov::Model>& outer_f) {
    const auto& params = sub_graph_op->get_function()->get_parameters();
    const auto& results = sub_graph_op->get_function()->get_results();
    // before: Layer1 -> TI [input -> bodyParameter -> Layer2 -> ...]
    // after:  Layer1 -> Layer2 ->...
    for (const auto& in : sub_graph_op->get_input_descriptions()) {
        const auto& connect_to = sub_graph_op->get_input_source_output(in->m_input_index);
        for (auto& output : params.at(in->m_body_parameter_index)->outputs()) {
            output.replace(connect_to);
        }
    }

    // before: TI [...-> Layer1 -> Result -> output] -> Layer2 -> ...
    // after:  ...-> Layer1 -> Layer2 -> ...
    ov::NodeVector new_ops;
    for (const auto& out : sub_graph_op->get_output_descriptions()) {
        const auto& connect_to = results.at(out->m_body_value_index)->get_input_source_output(0);
        for (auto& input_to : sub_graph_op->output(out->m_output_index).get_target_inputs()) {
            // create OV output name
            std::string out_name = sub_graph_op->get_friendly_name();
            if (sub_graph_op->get_output_size() != 1)
                out_name += "." + std::to_string(out->m_output_index);

            // IECompatibility: insert identity (Unsqueeze + Squeeze) to store the TensorIterator
            // output names
            auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
            auto identity_1 = std::make_shared<ov::op::v0::Unsqueeze>(connect_to, axis_1);
            auto identity_2 = std::make_shared<ov::op::v0::Squeeze>(identity_1, axis_1);
            identity_2->set_friendly_name(out_name);
            new_ops.push_back(identity_1);
            new_ops.push_back(identity_2);

            identity_2->output(0).get_tensor().add_names(input_to.get_source_output().get_names());
            input_to.replace_source_output(identity_2);
        }
    }
    outer_f->add_sinks(sub_graph_op->get_function()->get_sinks());
    ov::copy_runtime_info(sub_graph_op, sub_graph_op->get_function()->get_ops());
    ov::copy_runtime_info(sub_graph_op, std::move(new_ops));
}

ov::Output<ov::Node> create_init_subgraph(const ov::Output<ov::Node>& in_node, ov::pass::NodeRegistry& to) {
    auto const_zero = to.make<ov::op::v0::Constant>(in_node.get_element_type(), ov::Shape{1}, 0);
    auto shape_of = to.make<ov::op::v3::ShapeOf>(in_node);
    auto broadcast = to.make<ov::op::v3::Broadcast>(const_zero, shape_of);
    return broadcast->output(0);
}

std::shared_ptr<ov::op::v6::Assign> replace_with_memory(const ov::Input<ov::Node>& input,
                                                        const ov::Output<ov::Node>& output,
                                                        const std::string& variable_name,
                                                        bool use_const_initializer,
                                                        ov::pass::NodeRegistry& to) {
    using namespace ov::op::util;

    ov::Output<ov::Node> read_value_in = input.get_source_output();
    if (use_const_initializer) {
        read_value_in = create_init_subgraph(read_value_in, to);
    }

    VariableInfo var_info{read_value_in.get_partial_shape(), read_value_in.get_element_type(), variable_name};
    auto variable = std::make_shared<Variable>(var_info);
    auto read_value = to.make<ov::op::v6::ReadValue>(read_value_in, variable);
    input.replace_source_output(read_value->output(0));
    auto assign = to.make<ov::op::v6::Assign>(output, variable);
    // control dependency so that ReadValue is processed before Assign
    assign->add_control_dependency(read_value);
    return assign;
}

std::vector<std::shared_ptr<ov::op::v6::Assign>> replace_with_memory(const std::shared_ptr<ov::Node>& node,
                                                                     const std::vector<size_t>& indexes,
                                                                     bool use_const_initializer,
                                                                     ov::pass::NodeRegistry& to) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> new_assigns;
    size_t var_idx = 0;
    for (const auto& idx : indexes) {
        auto in = node->input(idx);
        auto out = node->output(idx);
        new_assigns.push_back(replace_with_memory(in,
                                                  out,
                                                  node->get_friendly_name() + "/variable_" + std::to_string(var_idx++),
                                                  use_const_initializer,
                                                  to));
    }
    return new_assigns;
}

bool need_unroll(const std::shared_ptr<ov::Node>& op) {
    const auto& p_shape = op->get_input_partial_shape(0);
    if (p_shape.rank().is_dynamic() || p_shape[1].is_dynamic() || p_shape[1].get_length() != 1) {
        return false;
    }
    return true;
}

ov::OutputVector prepare_inputs(const std::shared_ptr<ov::Node>& op, size_t seq_len_idx, ov::pass::NodeRegistry& to) {
    ov::OutputVector inputs;
    auto axis_0 = to.make<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, 0);
    auto axis_1 = to.make<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, 1);
    size_t num_lstm_inputs_without_peepholes = 7;
    for (size_t i = 0; i < std::min(op->get_input_size(), num_lstm_inputs_without_peepholes); ++i) {
        if (i < seq_len_idx) {
            inputs.push_back(to.make<ov::op::v0::Squeeze>(op->get_input_source_output(i), axis_1));
        } else if (i > seq_len_idx) {
            inputs.push_back(to.make<ov::op::v0::Squeeze>(op->get_input_source_output(i), axis_0));
        }
    }
    return inputs;
}

std::vector<std::shared_ptr<ov::op::v6::Assign>> process_sequence(const std::shared_ptr<ov::Node>& op,
                                                                  bool m_use_const_initializer,
                                                                  ov::pass::NodeRegistry& to) {
    std::shared_ptr<ov::Node> cell;
    std::vector<std::shared_ptr<ov::op::v6::Assign>> new_assigns;
    bool unroll = false;
    if (auto lstm_seq_v5 = ov::as_type_ptr<ov::op::v5::LSTMSequence>(op)) {
        unroll = need_unroll(op);
        new_assigns = replace_with_memory(op, {1, 2}, m_use_const_initializer, to);
        if (unroll) {
            auto inputs = prepare_inputs(op, 3, to);
            cell = to.make<ov::op::v4::LSTMCell>(inputs[0],
                                                 inputs[1],
                                                 inputs[2],
                                                 inputs[3],
                                                 inputs[4],
                                                 inputs[5],
                                                 lstm_seq_v5->get_hidden_size(),
                                                 lstm_seq_v5->get_activations(),
                                                 lstm_seq_v5->get_activations_alpha(),
                                                 lstm_seq_v5->get_activations_beta(),
                                                 lstm_seq_v5->get_clip());
        }
    } else if (auto gru_seq = ov::as_type_ptr<ov::op::v5::GRUSequence>(op)) {
        unroll = need_unroll(op);
        new_assigns = replace_with_memory(op, {1}, m_use_const_initializer, to);
        if (unroll) {
            auto inputs = prepare_inputs(op, 2, to);
            cell = to.make<ov::op::v3::GRUCell>(inputs[0],
                                                inputs[1],
                                                inputs[2],
                                                inputs[3],
                                                inputs[4],
                                                gru_seq->get_hidden_size(),
                                                gru_seq->get_activations(),
                                                gru_seq->get_activations_alpha(),
                                                gru_seq->get_activations_beta(),
                                                gru_seq->get_clip(),
                                                gru_seq->get_linear_before_reset());
        }
    } else if (auto rnn_seq = ov::as_type_ptr<ov::op::v5::RNNSequence>(op)) {
        unroll = need_unroll(op);
        new_assigns = replace_with_memory(op, {1}, m_use_const_initializer, to);
        if (unroll) {
            auto inputs = prepare_inputs(op, 2, to);
            cell = to.make<ov::op::v0::RNNCell>(inputs[0],
                                                inputs[1],
                                                inputs[2],
                                                inputs[3],
                                                inputs[4],
                                                rnn_seq->get_hidden_size(),
                                                rnn_seq->get_activations(),
                                                rnn_seq->get_activations_alpha(),
                                                rnn_seq->get_activations_beta(),
                                                rnn_seq->get_clip());
        }
    } else {
        // unsupported sequence or not sequence
        return {};
    }

    if (unroll && cell) {
        auto axis_1_2 = to.make<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{1, 2});
        auto axis_1 = to.make<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, 1);
        ov::OutputVector outputs;

        auto unsqueeze_Y = to.make<ov::op::v0::Unsqueeze>(cell->output(0), axis_1_2);
        unsqueeze_Y->set_friendly_name(op->get_friendly_name() + ":0");
        outputs.push_back(unsqueeze_Y);

        size_t idx = 1;
        for (const auto& out : cell->outputs()) {
            auto unsqueeze_state = to.make<ov::op::v0::Unsqueeze>(out, axis_1);
            unsqueeze_state->set_friendly_name(op->get_friendly_name() + ":" + std::to_string(idx++));
            outputs.push_back(unsqueeze_state);
        }
        replace_node(op, outputs);
    }
    copy_runtime_info(op, to.get());
    return new_assigns;
}
}  // namespace

bool ov::pass::LowLatency2::run_on_model(const std::shared_ptr<Model>& f) {
    RUN_ON_MODEL_SCOPE(LowLatency2);
    using namespace ov::op::util;
    NodeRegistry to;

    ov::SinkVector assigns;
    for (const auto& op : f->get_ordered_ops()) {
        ov::op::util::process_subgraph(*this, op);

        if (const auto& sub_graph_op = ov::as_type_ptr<SubGraphOp>(op)) {
            int64_t variable_id = 0;
            const auto& func = sub_graph_op->get_function();
            const auto& params = func->get_parameters();
            for (const auto& in : sub_graph_op->get_input_descriptions()) {
                // Process all back edges
                if (const auto& merged_in = ov::as_type_ptr<SubGraphOp::MergedInputDescription>(in)) {
                    // create new Variable
                    const std::string& param_name = params.at(merged_in->m_body_parameter_index)->get_friendly_name();
                    const std::string& var_name =
                        generate_variable_name(sub_graph_op->get_friendly_name(), param_name, variable_id);

                    const auto& input = sub_graph_op->input(merged_in->m_input_index);
                    if (ov::as_type_ptr<ReadValueBase>(input.get_source_output().get_node_shared_ptr()) != nullptr) {
                        OPENVINO_DEBUG(msg_low_latency_2_already_applied);
                        return false;
                    }

                    const auto& param =
                        sub_graph_op->get_function()->get_parameters().at(merged_in->m_body_parameter_index);
                    for (const auto& in_to : param->output(0).get_target_inputs()) {
                        if (ov::as_type<ReadValueBase>(in_to.get_node()) != nullptr) {
                            OPENVINO_DEBUG(msg_low_latency_already_applied);
                            return false;
                        }
                    }

                    /** insert ReadValue and Assign ops:
                     *
                     * Layers -> [new op: ReadValue] -> Subgraph operation
                     *
                     * Subgraph operation -> [new op: Assign]
                     *                    \
                     *                     ---> Layers -> ...
                     */
                    const auto& out_desc = sub_graph_op->get_output_descriptions();
                    bool is_output_exist =
                        any_of(out_desc.begin(),
                               out_desc.end(),
                               [&merged_in](const std::shared_ptr<SubGraphOp::OutputDescription>& out) {
                                   return out->m_body_value_index == merged_in->m_body_value_index;
                               });
                    // Create new output if it doesn't exist.
                    if (!is_output_exist) {
                        sub_graph_op->get_iter_value(func->get_results().at(merged_in->m_body_value_index));
                    }
                    Output<Node> output;
                    for (const auto& out : sub_graph_op->get_output_descriptions()) {
                        if (out->m_body_value_index == merged_in->m_body_value_index) {
                            output = sub_graph_op->output(out->m_output_index);
                            break;
                        }
                    }
                    auto assign = replace_with_memory(input, output, var_name, m_use_const_initializer, to);
                    assigns.emplace_back(assign);
                    copy_runtime_info(sub_graph_op, to.get());
                }

                variable_id++;
            }

            if (sub_graph_op->get_num_iterations() == 1) {
                unroll_single_iteration(sub_graph_op, f);
            }
        } else {
            auto new_assigns = process_sequence(op, m_use_const_initializer, to);
            if (!new_assigns.empty()) {
                assigns.insert(assigns.end(), new_assigns.begin(), new_assigns.end());
            }
        }
    }
    f->add_sinks(assigns);
    return true;
}
