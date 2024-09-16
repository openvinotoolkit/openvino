// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/control_flow/unroll_tensor_iterator.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

bool ov::pass::UnrollTensorIterator::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(UnrollTensorIterator);
    for (const auto& op : f->get_ops()) {
        ov::op::util::process_subgraph(*this, op);

        auto sub_graph_op = std::dynamic_pointer_cast<op::util::SubGraphOp>(op);
        if (!sub_graph_op || transformation_callback(sub_graph_op)) {
            continue;
        }

        const auto& function = sub_graph_op->get_function();
        int64_t num_iter = sub_graph_op->get_num_iterations();

        // negative value means inconsistent TI
        if (num_iter <= -1) {
            continue;
        }

        // Create copies of the TensorIterator body, the number of copies is equal to the number of iterations.
        // Assign names to the created layers.
        std::vector<std::shared_ptr<ov::Model>> body_functions(num_iter);
        for (int64_t idx = 0; idx < num_iter; ++idx) {
            body_functions[idx] = function->clone();
            for (auto& node : body_functions[idx]->get_ops()) {
                node->set_friendly_name(sub_graph_op->get_friendly_name() + "/" + std::to_string(idx + 1) + "/" +
                                        node->get_friendly_name());
                copy_runtime_info(sub_graph_op, node);
            }
        }

        // Port map : inputs and back edges
        for (const auto& desc : sub_graph_op->get_input_descriptions()) {
            if (const auto& input_desc =
                    std::dynamic_pointer_cast<ov::op::v0::TensorIterator::SliceInputDescription>(desc)) {
                // Connect the sliced input (layer before the input) to the Split layer and connect
                // the corresponding Split output to the corresponding copy of the body.
                // If the number of iterations is 1, then the Split is not needed.

                auto in_data = sub_graph_op->input_values()[input_desc->m_input_index];
                const auto const_axis = ov::op::v0::Constant::create(element::i64, Shape{}, {input_desc->m_axis});

                if (num_iter > 1) {
                    auto split = std::make_shared<ov::op::v1::Split>(in_data, const_axis, num_iter);
                    copy_runtime_info(sub_graph_op, split);
                    auto stride = input_desc->m_stride;
                    // connect to the body
                    for (int64_t j = 0; j < num_iter; j++) {
                        auto idx = stride > 0 ? j : num_iter - j - 1;
                        const auto& param = body_functions[j]->get_parameters()[input_desc->m_body_parameter_index];
                        for (auto& output : param->outputs()) {
                            output.replace(split->output(idx));
                        }
                    }
                } else {
                    // connect to the body
                    const auto& param = body_functions[0]->get_parameters()[input_desc->m_body_parameter_index];
                    for (auto& output : param->outputs()) {
                        output.replace(in_data);
                    }
                }
            } else if (const auto& merged_desc =
                           std::dynamic_pointer_cast<ov::op::v0::TensorIterator::MergedInputDescription>(desc)) {
                // Connect the input to the corresponding copy of the body.
                auto in_data = sub_graph_op->input_values()[merged_desc->m_input_index];
                const auto& param = body_functions[0]->get_parameters()[merged_desc->m_body_parameter_index];
                for (auto& output : param->outputs()) {
                    output.replace(in_data);
                }

                // Back-edge processing. Connect the copies of the body to each other.
                for (int64_t j = 1; j < num_iter; j++) {
                    const auto& cur_param = body_functions[j]->get_parameters()[merged_desc->m_body_parameter_index];
                    const auto& prev_val = body_functions[j - 1]->get_results()[merged_desc->m_body_value_index];
                    for (auto& output : cur_param->outputs()) {
                        output.replace(prev_val->get_input_source_output(0));
                    }
                }
            } else if (const auto& invariant_desc =
                           std::dynamic_pointer_cast<ov::op::v0::TensorIterator::InvariantInputDescription>(desc)) {
                // Connect the input to the corresponding copy of the body.
                auto in_data = sub_graph_op->input_values()[invariant_desc->m_input_index];
                for (int64_t j = 0; j < num_iter; j++) {
                    auto param = body_functions[j]->get_parameters()[invariant_desc->m_body_parameter_index];
                    for (auto& output : param->outputs()) {
                        output.replace(in_data);
                    }
                }
            } else {
                // "Incorrect type of the input description.";
                return false;
            }
        }

        // Port map: outputs
        for (const auto& desc : sub_graph_op->get_output_descriptions()) {
            //  we need to insert tensor_name to the outputs of TensorIterator if they directly connected to
            // Results ops. It's necessary to save original TensorIterator name when we use CNNNetwork.
            auto insert_tensor_name = [&](const ov::Output<ov::Node>& ti_output, const ov::Output<Node>& insert_to) {
                auto target_inputs = ti_output.get_target_inputs();
                if (target_inputs.empty() ||
                    std::any_of(target_inputs.begin(), target_inputs.end(), [](const ov::Input<ov::Node>& target_inp) {
                        return ov::as_type<ov::op::v0::Result>(target_inp.get_node()) != nullptr;
                    })) {
                    OPENVINO_SUPPRESS_DEPRECATED_START
                    ov::descriptor::set_ov_tensor_legacy_name(insert_to.get_tensor(),
                                                              ov::op::util::create_ie_output_name(ti_output));
                    OPENVINO_SUPPRESS_DEPRECATED_END
                }
            };

            if (const auto& concat_desc =
                    std::dynamic_pointer_cast<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc)) {
                if (!concat_desc) {
                    return false;
                }

                // Connect corresponding outputs (layers before Result op) of each copy of the body to Concat layer.
                // Connect the Concat to corresponding output of TensorIterator.
                // If the number of iterations is 1, then the Concat is not needed.

                if (num_iter > 1) {
                    ov::OutputVector to_concat(num_iter);
                    auto stride = concat_desc->m_stride;

                    // Connect outputs of the bodies to the Concat layer
                    for (int64_t j = 0; j < num_iter; j++) {
                        auto idx = stride > 0 ? j : num_iter - j - 1;
                        std::shared_ptr<ov::op::v0::Result> result =
                            body_functions[idx]->get_results()[concat_desc->m_body_value_index];
                        auto input_to_res = result->get_input_source_output(0);
                        to_concat[j] = input_to_res;
                    }
                    auto concat = std::make_shared<ov::op::v0::Concat>(to_concat, concat_desc->m_axis);
                    copy_runtime_info(sub_graph_op, concat);

                    // set output name to Tensor to store it for openvino to cnn conversion
                    insert_tensor_name(sub_graph_op->output(concat_desc->m_output_index), concat->output(0));

                    // connect the Concat layer to the corresponding TI outputs
                    for (auto& input : sub_graph_op->output(concat_desc->m_output_index).get_target_inputs()) {
                        input.replace_source_output(concat);
                    }
                } else {
                    // Connect outputs of the bodies to the corresponding TI outputs
                    std::shared_ptr<ov::op::v0::Result> result =
                        body_functions[0]->get_results().at(concat_desc->m_body_value_index);
                    const auto& input_to_res = result->get_input_source_output(0);
                    // set output name to Tensor to store it for openvino to cnn conversion
                    insert_tensor_name(sub_graph_op->output(concat_desc->m_output_index), input_to_res);

                    for (auto& input : sub_graph_op->output(concat_desc->m_output_index).get_target_inputs()) {
                        input.replace_source_output(input_to_res);
                    }
                }
            } else if (const auto& output_desc =
                           std::dynamic_pointer_cast<ov::op::v0::TensorIterator::BodyOutputDescription>(desc)) {
                // Connect outputs of the bodies to the corresponding TI outputs
                auto iter = output_desc->m_iteration;
                iter = iter >= 0 ? iter : num_iter - 1;
                std::shared_ptr<ov::op::v0::Result> result =
                    body_functions[iter]->get_results()[output_desc->m_body_value_index];
                const auto& in_value = result->input_value(0);

                insert_tensor_name(sub_graph_op->output(output_desc->m_output_index), in_value);
                for (const auto& input : sub_graph_op->output(output_desc->m_output_index).get_target_inputs()) {
                    input.replace_source_output(result->get_input_source_output(0));
                }
            } else {
                // "Incorrect type of the output description."
                return false;
            }
        }

        for (const auto& body_func : body_functions) {
            f->add_sinks(body_func->get_sinks());
        }

        // the current iteration Parameter in Loop body can be disconnected
        // we are replacing it with a Constant (value = current iteration idx)
        const auto& loop = ov::as_type_ptr<ov::op::v5::Loop>(sub_graph_op);
        if (loop) {
            // 1. Check CurrentIteration Parameter is not connected to outer network
            bool need_to_remove_iteration_param = false;
            const auto cur_iter_idx = loop->get_special_body_ports().current_iteration_input_idx;
            if (cur_iter_idx >= 0) {
                const auto& in_descs = loop->get_input_descriptions();
                need_to_remove_iteration_param =
                    std::all_of(in_descs.begin(),
                                in_descs.end(),
                                [cur_iter_idx](const std::shared_ptr<op::util::SubGraphOp::InputDescription>& in_desc) {
                                    return in_desc->m_body_parameter_index != static_cast<uint64_t>(cur_iter_idx);
                                });
            }

            // 2. Replace CurrentIteration Parameter with a Constant for each copy of the body
            if (need_to_remove_iteration_param) {
                for (int64_t idx = 0; idx < num_iter; ++idx) {
                    const auto iter_idx = loop->get_special_body_ports().current_iteration_input_idx;
                    const auto& param_to_delete = body_functions[idx]->get_parameters()[iter_idx];
                    auto cur_iter_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, Shape{}, idx);
                    replace_node(param_to_delete, cur_iter_const);
                    body_functions[idx]->remove_parameter(param_to_delete);
                }
            }
        }
    }
    return true;
}
