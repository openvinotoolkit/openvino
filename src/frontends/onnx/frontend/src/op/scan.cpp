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

    const int64_t num_scan_inputs = node.get_attribute_value<int64_t>("num_scan_inputs");
    const std::vector<int64_t> default_values(num_scan_inputs, 0);
    std::vector<int64_t> scan_input_axes = node.get_attribute_value<std::vector<int64_t>>("scan_input_axes", default_values);
    std::vector<int64_t> scan_input_directions = node.get_attribute_value<std::vector<int64_t>>("scan_input_directions", default_values);
    std::vector<int64_t> scan_output_axes = node.get_attribute_value<std::vector<int64_t>>("scan_output_axes", default_values);
    std::vector<int64_t> scan_output_directions = node.get_attribute_value<std::vector<int64_t>>("scan_output_directions", default_values);

    // N initial values N state,
    // const auto init_val_inputs_count = ng_inputs.size() - num_scan_inputs;
    const auto init_val_inputs_count = 1;

    const OutputVector init_i{ng_inputs.begin(), ng_inputs.begin() + init_val_inputs_count}; // initial
    const OutputVector scan_i{ng_inputs.begin() + init_val_inputs_count, ng_inputs.end()}; // x

    const auto& subgraphs = node.get_subgraphs();
    auto body_graph = subgraphs.at("body"); // Add(sum_in, next)
    auto body_outputs = body_graph->get_ng_outputs(); // (2) Add(sum_in, next) -> sum_out, Identity->scan_out
    const auto& body_inputs = body_graph->get_ng_parameters(); // (2) (sum_in, next)

    auto initial = ng_inputs[0];
    auto x = ng_inputs[1];

    auto sum_in = body_inputs[0];
    auto next = body_inputs[1];

    auto sum_out = body_outputs[0];
    auto scan_out = body_outputs[1];

    // // Infer body inputs' element type based on carried dependencies
    for (size_t i = 0; i < init_i.size(); i++) { // sum_in at 0, by init
        body_inputs[i]->set_element_type(init_i[i].get_element_type());
        body_inputs[i]->set_partial_shape(init_i[i].get_partial_shape());
    }

    //  sequence_length = scan_1.shape[axis_1];
    // auto sequence_length = scan_i[0].get_shape()[scan_input_axes[0]]; // TODO: Update to dynamic PartialShape
    // auto sequence_length = x.get_shape()[scan_input_axes[0]]; // TODO: Update to dynamic PartialShape
    auto sequence_length = 3; // Test value

    Output<ngraph::Node> trip_count = ngraph::op::Constant::create(ngraph::element::i64, {1}, {sequence_length});
    Output<ngraph::Node> termination_cond = ngraph::op::Constant::create(ngraph::element::boolean, {1}, {true});

    // OV Concat op related
    const int64_t concat_axis = 0;
    const auto concat_axis_const = ngraph::op::Constant::create(ngraph::element::i64, {1}, {concat_axis});
    // add dimension along which scan outputs will be concatenated
    for (size_t i = 1; i < body_outputs.size(); ++i) {
        body_outputs[i] = std::make_shared<default_opset::Unsqueeze>(body_outputs[i], concat_axis_const);
    }

    scan_out = std::make_shared<default_opset::Unsqueeze>(scan_out, concat_axis_const);

    auto body_condition = std::make_shared<default_opset::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);
    auto current_iteration = std::make_shared<default_opset::Parameter>(element::i64, Shape{});

    //// Trying Slice op, (poc usage instead of set_sliced_input)
    // const auto start = std::make_shared<default_opset::Unsqueeze>(current_iteration, concat_axis_const);
    // const auto step = ngraph::op::Constant::create(ngraph::element::i64, {1}, {1});
    // const auto stop = std::make_shared<default_opset::Add>(current_iteration, step);
    // const auto axis = ngraph::op::Constant::create(ngraph::element::i64, {1}, {0});

    // // auto sliced_next_param = std::make_shared<default_opset::Parameter>(element::i64, Shape{2});
    // auto sliced_next = std::make_shared<default_opset::Slice>(x, start, stop, step, axis);
    // auto squeezed_slice = std::make_shared<default_opset::Squeeze>(sliced_next, axis);

    ////
    // ParameterVector body_params{sum_in, next};
    // body_params.emplace(body_params.begin(), current_iteration);  // current iteration body input
    // body_outputs.emplace(body_outputs.begin(), body_condition);

    // const auto body = std::make_shared<ngraph::Function>(body_outputs,
    //                                                     body_params);


    const auto body = std::make_shared<ngraph::Function>(OutputVector{body_condition, sum_out, scan_out},
                                                        ParameterVector{current_iteration, sum_in, next});

    auto loop = std::make_shared<default_opset::Loop>(trip_count, termination_cond);
    default_opset::Loop::SpecialBodyPorts spec_ports{0, 0}; // current_iter_input idx, body_condidiotn_input idx

    loop->set_special_body_ports(spec_ports);
    loop->set_function(body);

    OutputVector final_values;
    loop->set_merged_input(sum_in, initial, sum_out);

    // Slice of x input per iteration, (RuntimeError: Could not create a primitive descriptor for a reorder primitive)
    // Without, set_sliced_input all results equal 0
    loop->set_sliced_input(next, x, 0, 1, 1, -1, scan_input_axes[0]);

    //// another tests
    // loop->set_merged_input(next, squeezed_slice, squeezed_slice);
    // loop->set_invariant_input(next, squeezed_slice);
    // loop->set_invariant_input(next, x);

    final_values.push_back(loop->get_iter_value(sum_out, -1));

    loop->validate_and_infer_types();

    auto y = loop->get_iter_value(sum_out, -1); // y final value
    auto z = loop->get_concatenated_slices(scan_out, 0, 1, 1, -1, concat_axis);

    return OutputVector{y, z};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
