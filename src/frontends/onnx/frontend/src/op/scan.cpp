// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/scan.hpp"

#include <iterator>
#include <memory>

#include "core/graph.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "onnx_import/core/null_node.hpp"
#include "openvino/core/validation_util.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {

namespace {

OutputVector scan_to_tensor_iterator(const OutputVector& node_inputs,
                                     ParameterVector& body_inputs,
                                     OutputVector& body_outputs,
                                     int64_t num_scan_inputs,
                                     const std::vector<int64_t>& scan_input_axes,
                                     const std::vector<int64_t>& scan_input_directions,
                                     const std::vector<int64_t>& scan_output_axes,
                                     const std::vector<int64_t>& scan_output_directions,
                                     int64_t in_offset = 0,
                                     const std::string& node_description = "") {
    const size_t num_initial_values = body_inputs.size() - num_scan_inputs;
    const size_t num_scan_outputs = body_outputs.size() - num_initial_values;

    // Body inputs alignment
    for (size_t i = 0; i < num_initial_values; ++i) {
        body_inputs[i]->set_element_type(node_inputs[i + in_offset].get_element_type());
        body_inputs[i]->set_partial_shape(node_inputs[i + in_offset].get_partial_shape());
        body_inputs[i]->validate_and_infer_types();
    }
    // Single slice of TensorIterator sliced input has the same rank as the input,
    // but in ONNX Scan the slice of input can has one dimension less,
    // so the parameter needs to have aligned rank with 1 at sliced axis,
    // and then squeezed to restore original shape.
    for (int64_t i = 0; i < num_scan_inputs; ++i) {
        const auto in_idx = num_initial_values + i;
        auto axis = scan_input_axes[i];
        const auto axis_node = default_opset::Constant::create(element::i64, Shape{1}, {axis});
        auto shape = node_inputs[in_idx + in_offset].get_partial_shape();
        if (shape.rank().is_static()) {
            axis = ov::normalize_axis(node_description,
                                      scan_input_axes[i],
                                      node_inputs[in_idx + in_offset].get_partial_shape().rank());
            shape[axis] = 1;
        }
        body_inputs[in_idx]->set_partial_shape(shape);
        body_inputs[in_idx]->validate_and_infer_types();

        auto input_consumers = body_inputs[in_idx]->output(0).get_target_inputs();
        auto squeeze = std::make_shared<default_opset::Squeeze>(body_inputs[in_idx], axis_node);
        for (auto& input : input_consumers) {
            input.replace_source_output(squeeze);
        }
    }
    // Body outputs shape alignment, add dimension along which scan outputs will be concatenated
    for (size_t i = 0; i < num_scan_outputs; ++i) {
        const auto out_idx = num_initial_values + i;
        const auto axis = scan_output_axes[i];
        const auto axis_node = default_opset::Constant::create(element::i64, Shape{1}, {axis});
        body_outputs[out_idx] = std::make_shared<default_opset::Unsqueeze>(body_outputs[out_idx], axis_node);
    }

    // TensorIterator setup
    auto tensor_iterator = std::make_shared<default_opset::TensorIterator>();
    auto ti_body = std::make_shared<ov::Model>(body_outputs, body_inputs);
    tensor_iterator->set_function(ti_body);

    // Set slicing for Scan (TensorIterator) inputs
    for (int64_t i = 0; i < num_scan_inputs; ++i) {
        const auto in_idx = num_initial_values + i;
        const auto axis = ov::normalize_axis(node_description,
                                             scan_input_axes[i],
                                             node_inputs[in_idx + in_offset].get_partial_shape().rank());
        if (scan_input_directions[i]) {  // reverse direction
            tensor_iterator->set_sliced_input(body_inputs[in_idx], node_inputs[in_idx + in_offset], -1, -1, 1, 0, axis);
        } else {  // forward direction
            tensor_iterator->set_sliced_input(body_inputs[in_idx], node_inputs[in_idx + in_offset], 0, 1, 1, -1, axis);
        }
    }

    // Set Scan (TensorIterator) outputs
    OutputVector outputs;
    for (size_t i = 0; i < num_initial_values; ++i) {
        // Back edge for state input/output
        tensor_iterator->set_merged_input(body_inputs[i], node_inputs[i + in_offset], body_outputs[i]);
        outputs.push_back(tensor_iterator->get_iter_value(body_outputs[i], -1));
    }
    for (size_t i = 0; i < num_scan_outputs; ++i) {
        const auto out_idx = num_initial_values + i;
        const auto axis =
            ov::normalize_axis(node_description, scan_output_axes[i], body_outputs[out_idx].get_partial_shape().rank());
        if (scan_output_directions[i]) {  // reverse direction
            outputs.push_back(tensor_iterator->get_concatenated_slices(body_outputs[out_idx], -1, -1, 1, 0, axis));
        } else {  // forward direction
            outputs.push_back(tensor_iterator->get_concatenated_slices(body_outputs[out_idx], 0, 1, 1, -1, axis));
        }
    }

    return outputs;
}

OutputVector import_onnx_scan(const Node& node,
                              int64_t default_axis,
                              int64_t in_offset,
                              std::string&& in_directions_attr_name) {
    const auto& node_inputs = node.get_ng_inputs();

    const auto& subgraphs = node.get_subgraphs();
    auto body_graph = subgraphs.at("body");
    auto body_outputs = body_graph->get_ng_outputs();
    auto body_inputs = body_graph->get_ng_parameters();

    const int64_t num_scan_inputs = node.get_attribute_value<int64_t>("num_scan_inputs");
    const size_t num_initial_values = body_inputs.size() - num_scan_inputs;
    const size_t num_scan_outputs = body_outputs.size() - num_initial_values;

    std::vector<int64_t> scan_input_axes =
        node.get_attribute_value<std::vector<int64_t>>("scan_input_axes",
                                                       std::vector<int64_t>(num_scan_inputs, default_axis));
    std::vector<int64_t> scan_input_directions =
        node.get_attribute_value<std::vector<int64_t>>(in_directions_attr_name,
                                                       std::vector<int64_t>(num_scan_inputs, 0));
    std::vector<int64_t> scan_output_axes =
        node.get_attribute_value<std::vector<int64_t>>("scan_output_axes",
                                                       std::vector<int64_t>(num_scan_outputs, default_axis));
    std::vector<int64_t> scan_output_directions =
        node.get_attribute_value<std::vector<int64_t>>("scan_output_directions",
                                                       std::vector<int64_t>(num_scan_outputs, 0));

    return scan_to_tensor_iterator(node_inputs,
                                   body_inputs,
                                   body_outputs,
                                   num_scan_inputs,
                                   scan_input_axes,
                                   scan_input_directions,
                                   scan_output_axes,
                                   scan_output_directions,
                                   in_offset,
                                   node.get_description());
}

}  // namespace

namespace set_1 {

OutputVector scan(const Node& node) {
    // ONNX Scan-8 can have optional `sequence_lens` input,
    // and sequence scan_input axis is assumed to be always 1.
    OPENVINO_ASSERT(ngraph::op::is_null(node.get_ng_inputs().at(0)),
                    node.get_description(),
                    " ONNX Scan-8 `sequence_lens` input is not supported. ");
    return import_onnx_scan(node, 1, 1, "directions");
}

}  // namespace set_1

namespace set_9 {

OutputVector scan(const Node& node) {
    // Since ONNX Scan-9 the optional `sequence_lens input` was removed,
    // new attributes to specify input/output axes and directions were added.
    return import_onnx_scan(node, 0, 0, "scan_input_directions");
}

}  // namespace set_9
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
