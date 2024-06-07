// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/loop.hpp"

#include <algorithm>

#include "openvino/core/node.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/reference/concat.hpp"
#include "openvino/reference/function.hpp"
#include "openvino/reference/split.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace reference {
void loop(const std::shared_ptr<Model>& func,
          const op::util::OutputDescriptionVector& out_descs,
          const op::util::InputDescriptionVector& input_descs,
          const op::v5::Loop::SpecialBodyPorts& special_ports,
          ov::TensorVector& out,
          const ov::TensorVector& args,
          const EvaluationContext& evaluation_context) {
    const auto& cur_iter_idx = special_ports.current_iteration_input_idx;
    auto val = std::find_if(input_descs.begin(),
                            input_descs.end(),
                            [&cur_iter_idx](const op::util::SubGraphOp::InputDescription::Ptr& in_desc) {
                                return in_desc->m_body_parameter_index == static_cast<uint64_t>(cur_iter_idx);
                            });
    bool cur_iter_initial_value_exist = val != input_descs.end();
    bool cur_iter_back_edge_exist = false;

    // If current_iteration_input is exist and initial value is not provided, we
    // should allocate input_descs.size() + 1 inputs and set default value (0) for
    // current_iteration input.
    int64_t inputs_count = input_descs.size() + (cur_iter_idx >= 0 ? !cur_iter_initial_value_exist : 0);
    ov::TensorVector inputs_to_body;
    for (int64_t i = 0; i < inputs_count; ++i)
        inputs_to_body.push_back(ov::Tensor());
    if (cur_iter_idx >= 0 && !cur_iter_initial_value_exist) {
        const auto& cur_iter = func->get_parameters().at(cur_iter_idx);
        if (cur_iter->get_partial_shape().is_dynamic()) {
            cur_iter->set_partial_shape(Shape{1});
            cur_iter->validate_and_infer_types();
        }

        auto init = std::make_shared<op::v0::Constant>(func->get_parameters().at(cur_iter_idx)->get_element_type(),
                                                       func->get_parameters().at(cur_iter_idx)->get_shape(),
                                                       0);
        ov::Tensor in_tensor(func->get_parameters().at(cur_iter_idx)->get_element_type(),
                             func->get_parameters().at(cur_iter_idx)->get_shape());
        std::memset(in_tensor.data(), 0, in_tensor.get_byte_size());
        inputs_to_body.at(cur_iter_idx) = std::move(in_tensor);
    }

    // Port map processing: inputs and back edges
    struct BackEdge {
        uint64_t param_idx;
        uint64_t result_idx;
    };
    std::vector<BackEdge> back_edges;
    for (const auto& desc : input_descs) {
        inputs_to_body[desc->m_body_parameter_index] = args[desc->m_input_index];
        if (const auto& merged_desc = std::dynamic_pointer_cast<op::v5::Loop::MergedInputDescription>(desc)) {
            back_edges.push_back({merged_desc->m_body_parameter_index, merged_desc->m_body_value_index});
            cur_iter_back_edge_exist |= merged_desc->m_body_parameter_index == static_cast<uint64_t>(cur_iter_idx);
        }
    }

    // Get TripCount
    int64_t trip_count = 0;
    if (args[0].get_element_type() == ov::element::i32) {
        auto* trip_count_p = args[0].data<int32_t>();
        trip_count = trip_count_p[0];
    } else if (args[0].get_element_type() == ov::element::i64) {
        auto* trip_count_p = args[0].data<int64_t>();
        trip_count = trip_count_p[0];
    } else {
        OPENVINO_THROW("Unsupported element type for trip_count input. Expected int32 or int64.");
    }
    OPENVINO_ASSERT(trip_count != 0, "Zero count of iteration not supported");

    // Loop iterations
    auto exec_condition = args[1].data<bool>();
    if (exec_condition[0]) {
        // Find all ConcatOutputDescription
        std::vector<std::shared_ptr<op::v5::Loop::ConcatOutputDescription>> concat_outputs;
        for (const auto& desc : out_descs) {
            if (const auto& concat_desc = std::dynamic_pointer_cast<op::v5::Loop::ConcatOutputDescription>(desc)) {
                concat_outputs.push_back(concat_desc);
            }
        }

        // Slicing
        std::vector<std::shared_ptr<op::v0::TensorIterator::SliceInputDescription>> slice_inputs;
        std::vector<ov::TensorVector> sliced_values;
        int slice_in_idx = 0;
        for (const auto& desc : input_descs) {
            if (const auto& slice_desc =
                    std::dynamic_pointer_cast<op::v0::TensorIterator::SliceInputDescription>(desc)) {
                const auto el_size = args[slice_desc->m_input_index].get_element_type().size();
                slice_inputs.push_back(slice_desc);
                auto shape = args[slice_desc->m_input_index].get_shape();
                uint64_t num_iterations = shape.at(slice_desc->m_axis);
                shape.at(slice_desc->m_axis) = 1;
                sliced_values.emplace_back(ov::TensorVector());
                for (uint64_t i = 0; i < num_iterations; ++i) {
                    sliced_values.back().emplace_back(
                        ov::Tensor(args[slice_desc->m_input_index].get_element_type(), shape));
                }
                std::vector<char*> pointers_to_data(num_iterations);
                for (size_t j = 0; j < pointers_to_data.size(); ++j) {
                    pointers_to_data[slice_desc->m_stride > 0 ? j : (pointers_to_data.size() - j - 1)] =
                        static_cast<char*>(sliced_values[slice_in_idx][j].data());
                }
                reference::split(static_cast<char*>(args[slice_desc->m_input_index].data()),
                                 args[slice_desc->m_input_index].get_shape(),
                                 el_size,
                                 slice_desc->m_axis,
                                 num_iterations,
                                 pointers_to_data.data());
                slice_in_idx++;
            }
        }

        // Allocate vectors for store output values
        std::vector<ov::TensorVector> values_to_concat(concat_outputs.size());
        ov::TensorVector body_outputs;

        // Negative value means infinity count of iterations
        trip_count = trip_count >= 0 ? trip_count : std::numeric_limits<int64_t>::max();
        for (int64_t cur_iter = 0; cur_iter < trip_count; ++cur_iter) {
            // Copy new values for sliced inputs
            for (size_t i = 0; i < slice_inputs.size(); ++i) {
                if (static_cast<int64_t>(sliced_values[i].size()) > cur_iter)
                    inputs_to_body[slice_inputs[i]->m_body_parameter_index] = sliced_values[i][cur_iter];
            }

            // Evaluate body
            body_outputs.clear();
            reference::function(func, inputs_to_body, body_outputs, evaluation_context);

            // Store values for later concatenation
            for (size_t i = 0; i < values_to_concat.size(); ++i) {
                values_to_concat[i].push_back(body_outputs[concat_outputs[i]->m_body_value_index]);
            }

            // Check execution condition
            bool body_exec_condition(false);
            if (static_cast<int64_t>(body_outputs.size()) > special_ports.body_condition_output_idx &&
                body_outputs[special_ports.body_condition_output_idx])
                body_exec_condition = body_outputs[special_ports.body_condition_output_idx].data<bool>()[0];
            if (!body_exec_condition)
                break;

            // If there are no rules for calculating the current iteration, just
            // increment it.
            if (cur_iter_idx >= 0 && !cur_iter_back_edge_exist) {
                const auto& cur_iter_param = func->get_parameters().at(cur_iter_idx);
                int64_t iter_num = cur_iter + 1;
                if (cur_iter_param->get_element_type() == element::i64)
                    std::memcpy(inputs_to_body.at(cur_iter_idx).data(),
                                &iter_num,
                                cur_iter_param->get_element_type().size());
                else if (cur_iter_param->get_element_type() == element::i32) {
                    int32_t iter_num_i32 = static_cast<int32_t>(iter_num);
                    std::memcpy(inputs_to_body.at(cur_iter_idx).data(),
                                &iter_num_i32,
                                cur_iter_param->get_element_type().size());
                } else
                    OPENVINO_THROW("Unsupported element type for current iteration "
                                   "input. Expected int32 or int64.");
            }

            // Back-edge processing
            bool need_validate = false;
            for (auto& back_edge : back_edges) {
                const auto& input_shape = inputs_to_body[back_edge.param_idx].get_shape();
                const auto& result_shape = body_outputs[back_edge.result_idx].get_shape();
                // when output shape does not equal to input shape in a back-edge, such as
                //          Parameter(out:1)->|
                //                            |->Concat(out:2)->Result(out:2)
                //              Const(out:1)->|
                // after iteration completed, should update (out:2) to input, then use new input
                // shape to propagate others.
                if (input_shape != result_shape) {
                    const auto& param = func->get_parameters().at(back_edge.param_idx);
                    param->set_partial_shape(result_shape);
                    need_validate = true;
                }
                inputs_to_body[back_edge.param_idx] = body_outputs[back_edge.result_idx];
            }
            if (need_validate)
                func->validate_nodes_and_infer_types();
        }

        for (const auto& desc : out_descs) {
            if (const auto& body_desc = std::dynamic_pointer_cast<op::v5::Loop::BodyOutputDescription>(desc)) {
                const auto& res = body_outputs[body_desc->m_body_value_index];
                res.copy_to(out[body_desc->m_output_index]);
            }
        }

        // Concatenate and copy all values stored in values_to_concat vector to outputs
        for (size_t i = 0; i < concat_outputs.size(); ++i) {
            const auto& concat_desc = concat_outputs[i];
            auto shape = func->get_results().at(concat_desc->m_body_value_index)->get_shape();
            std::vector<Shape> shapes_to_concat(values_to_concat[i].size(), shape);
            shape.at(concat_desc->m_axis) = values_to_concat[i].size();
            out[concat_desc->m_output_index].set_shape(shape);
            std::vector<const char*> pointers_on_values;
            pointers_on_values.reserve(values_to_concat[i].size());
            for (const auto& vec : values_to_concat[i]) {
                pointers_on_values.push_back(static_cast<char*>(vec.data()));
            }
            reference::concat(pointers_on_values,
                              static_cast<char*>(out[concat_desc->m_output_index].data()),
                              shapes_to_concat,
                              shape,
                              concat_desc->m_axis,
                              out[concat_desc->m_output_index].get_element_type().size());
        }
    } else {
        OPENVINO_THROW("ExecutionCondition is false. Zero count of iteration not supported.");
    }
}
}  // namespace reference
}  // namespace ov
