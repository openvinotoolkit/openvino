// Copyright (C) 2018-2022 Intel Corporation
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
#include "utils/reshape.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector scan(const Node& node) {
    const auto& ng_inputs = node.get_ng_inputs();

    const auto& subgraphs = node.get_subgraphs();
    auto body_graph = subgraphs.at("body");
    auto body_outputs = body_graph->get_ng_outputs();
    auto body_inputs = body_graph->get_ng_parameters();

    const int64_t num_scan_inputs = node.get_attribute_value<int64_t>("num_scan_inputs");
    const size_t num_initial_values = ng_inputs.size() - num_scan_inputs;
    const size_t num_scan_outputs = body_outputs.size() - num_initial_values;

    std::vector<int64_t> scan_input_axes =
        node.get_attribute_value<std::vector<int64_t>>("scan_input_axes", std::vector<int64_t>(num_scan_inputs, 0));
    std::vector<int64_t> scan_input_directions =
        node.get_attribute_value<std::vector<int64_t>>("scan_input_directions",
                                                       std::vector<int64_t>(num_scan_inputs, 0));
    std::vector<int64_t> scan_output_axes =
        node.get_attribute_value<std::vector<int64_t>>("scan_output_axes", std::vector<int64_t>(num_scan_outputs, 0));
    std::vector<int64_t> scan_output_directions =
        node.get_attribute_value<std::vector<int64_t>>("scan_output_directions",
                                                       std::vector<int64_t>(num_scan_outputs, 0));

    for (size_t i = 0; i < body_inputs.size(); i++) {
        body_inputs[i]->set_element_type(ng_inputs[i].get_element_type());
        if (i < num_initial_values) {
            body_inputs[i]->set_partial_shape(ng_inputs[i].get_partial_shape());
            body_inputs[i]->validate_and_infer_types();
            continue;
        }

        auto axis_val = scan_input_axes[i - num_initial_values];
        auto shape = ng_inputs[i].get_partial_shape();
        if (shape.rank().is_static()) {
            shape[axis_val] = 1;
        }
        body_inputs[i]->set_partial_shape(shape);
        body_inputs[i]->validate_and_infer_types();

        auto input_consumers = body_inputs[i]->output(0).get_target_inputs();
        auto axis = default_opset::Constant::create(element::i64, Shape{1}, {axis_val});
        auto squeeze = std::make_shared<default_opset::Squeeze>(body_inputs[i], axis);
        for (auto& input : input_consumers) {
            input.replace_source_output(squeeze);
        }
    }

    for (size_t i = num_initial_values; i < body_outputs.size(); i++) {
        auto axis_val = scan_output_axes[i - num_initial_values];
        auto axis = default_opset::Constant::create(element::i64, Shape{1}, {axis_val});
        body_outputs[i] = std::make_shared<default_opset::Unsqueeze>(body_outputs[i], axis);
    }

    auto ti_body = std::make_shared<Function>(body_outputs, body_inputs);

    auto tensor_iterator = std::make_shared<default_opset::TensorIterator>();
    tensor_iterator->set_function(ti_body);

    OutputVector outputs;
    const auto& ti_body_results = ti_body->get_results();
    for (size_t i = 0; i < num_initial_values; i++) {
        tensor_iterator->set_merged_input(body_inputs[i], ng_inputs[i], ti_body_results[i]);
        outputs.push_back(tensor_iterator->get_iter_value(ti_body_results[i], -1));
    }

    for (size_t i = num_initial_values; i < num_initial_values + num_scan_inputs; i++) {
        auto direction = scan_input_directions[i - num_initial_values];
        auto axis = scan_input_axes[i - num_initial_values];
        if (direction == 0) {
            tensor_iterator->set_sliced_input(body_inputs[i], ng_inputs[i], 0, 1, 1, -1, axis);
        } else {
            tensor_iterator->set_sliced_input(body_inputs[i], ng_inputs[i], -1, -1, 1, 0, axis);
        }
    }

    for (size_t i = num_initial_values; i < ti_body_results.size(); i++) {
        auto direction = scan_output_directions[i - num_initial_values];
        auto axis = scan_output_axes[i - num_initial_values];
        if (direction == 0) {
            outputs.push_back(tensor_iterator->get_concatenated_slices(ti_body_results[i], 0, 1, 1, -1, axis));
        } else {
            outputs.push_back(tensor_iterator->get_concatenated_slices(ti_body_results[i], -1, -1, 1, 0, axis));
        }
    }

    return outputs;
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
