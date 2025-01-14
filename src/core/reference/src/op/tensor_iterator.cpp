// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/tensor_iterator.hpp"

#include "openvino/op/loop.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/reference/concat.hpp"
#include "openvino/reference/function.hpp"
#include "openvino/reference/split.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace reference {
void tensor_iterator(uint64_t num_iterations,
                     const std::shared_ptr<Model>& func,
                     const op::util::OutputDescriptionVector& out_descs,
                     const op::util::InputDescriptionVector& input_descs,
                     ov::TensorVector& out,
                     const ov::TensorVector& args,
                     const custom_evaluate_function& evaluate) {
    ov::TensorVector inputs_to_body;
    for (size_t i = 0; i < input_descs.size(); ++i)
        inputs_to_body.push_back(ov::Tensor());

    // Port map processing: inputs and back edges
    struct BackEdge {
        uint64_t param_idx;
        uint64_t result_idx;
    };
    std::vector<BackEdge> back_edges;
    for (const auto& desc : input_descs) {
        inputs_to_body[desc->m_body_parameter_index] = args[desc->m_input_index];
        if (const auto& merged_desc = ov::as_type_ptr<ov::op::v5::Loop::MergedInputDescription>(desc)) {
            back_edges.push_back({merged_desc->m_body_parameter_index, merged_desc->m_body_value_index});
        }
    }
    // Find all ConcatOutputDescription
    std::vector<std::shared_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>> concat_outputs;
    for (const auto& desc : out_descs) {
        if (const auto& concat_desc = ov::as_type_ptr<op::v0::TensorIterator::ConcatOutputDescription>(desc)) {
            concat_outputs.push_back(concat_desc);
        }
    }

    // Slicing
    std::vector<std::shared_ptr<op::v0::TensorIterator::SliceInputDescription>> slice_inputs;
    std::vector<ov::TensorVector> sliced_values;
    int slice_in_idx = 0;
    for (const auto& desc : input_descs) {
        if (const auto& slice_desc = ov::as_type_ptr<op::v0::TensorIterator::SliceInputDescription>(desc)) {
            const auto el_size = args[slice_desc->m_input_index].get_element_type().size();
            slice_inputs.push_back(slice_desc);
            auto shape = args[slice_desc->m_input_index].get_shape();
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

    for (uint64_t cur_iter = 0; cur_iter < num_iterations; ++cur_iter) {
        // Copy new values for sliced inputs
        for (size_t i = 0; i < slice_inputs.size(); ++i) {
            if (sliced_values[i].size() > cur_iter)
                inputs_to_body[slice_inputs[i]->m_body_parameter_index] = sliced_values[i][cur_iter];
        }

        // Evaluate body
        body_outputs.clear();
        if (!evaluate) {
            reference::function(func, inputs_to_body, body_outputs);
        } else {
            evaluate(func, inputs_to_body, body_outputs);
        }

        // Store values for later concatenation
        for (size_t i = 0; i < values_to_concat.size(); ++i) {
            values_to_concat[i].push_back(body_outputs[concat_outputs[i]->m_body_value_index]);
        }

        // Back-edge processing
        for (auto& back_edge : back_edges) {
            inputs_to_body[back_edge.param_idx] = body_outputs[back_edge.result_idx];
        }
    }

    for (const auto& desc : out_descs) {
        if (const auto& body_desc = ov::as_type_ptr<op::v0::TensorIterator::BodyOutputDescription>(desc)) {
            // Copy output values from the last iteration
            const auto& res = body_outputs[body_desc->m_body_value_index];
            res.copy_to(out[body_desc->m_output_index]);
        }
    }

    // Concatenate and copy all values stored in values_to_concat vector to outputs
    for (size_t i = 0; i < concat_outputs.size(); ++i) {
        const auto& concat_desc = concat_outputs[i];
        if (!concat_desc)
            continue;
        auto shape = func->get_results().at(concat_desc->m_body_value_index)->get_shape();
        std::vector<Shape> shapes_to_concat(values_to_concat[i].size(), shape);
        shape.at(concat_desc->m_axis) = values_to_concat[i].size();
        out[concat_desc->m_output_index].set_shape(shape);
        std::vector<const char*> pointers_on_values;
        pointers_on_values.reserve(values_to_concat[i].size());
        for (size_t j = 0; j < values_to_concat[i].size(); ++j) {
            size_t idx = concat_desc->m_stride > 0 ? j : (values_to_concat[i].size() - j - 1);
            if (values_to_concat[i].size() > idx && values_to_concat[i][idx])
                pointers_on_values.push_back(static_cast<char*>(values_to_concat[i][idx].data()));
        }
        reference::concat(pointers_on_values,
                          static_cast<char*>(out[concat_desc->m_output_index].data()),
                          shapes_to_concat,
                          shape,
                          concat_desc->m_axis,
                          out[concat_desc->m_output_index].get_element_type().size());
    }
}
}  // namespace reference
}  // namespace ov
