// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/loop.hpp"

#include <limits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/opsets/opset10.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_loop(const NodeContext& context) {
    const auto& inputs = context.inputs();
    PYTORCH_OP_CONVERSION_CHECK(inputs.size() >= 2, "Loop must have at least 2 inputs.");
    auto loop = std::make_shared<ov::op::v5::Loop>(inputs[0], inputs[1]);
    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 1, "Loop must have 1 subgraph.");
    auto subgraph_decoder = decoder->get_subgraph_decoder(0);
    auto body = context.convert_subgraph(0);
    loop->set_function(body);
    ov::op::v5::Loop::SpecialBodyPorts spec_ports{0, 0};
    loop->set_special_body_ports(spec_ports);

    // process outputs first
    auto session = context.get_session();
    auto body_results = body->get_results();
    PYTORCH_OP_CONVERSION_CHECK(body_results.size() > 0, "At least one output from loop is required - condition.");
    std::map<size_t, Output<Node>> output_idxs;
    // 0 output is condition, do not need to connect it
    for (size_t i = 1; i < body_results.size(); i++) {
        auto result = body_results[i];
        auto out_idx = session->decode_tensor_name(result->input(0).get_source_output());
        PYTORCH_OP_CONVERSION_CHECK(output_idxs.count(out_idx) == 0,
                                    "More than one body output with same tensor name.");
        output_idxs[out_idx] = result;
    }

    auto body_parameters = body->get_parameters();
    // #0 body parameter is counter;
    PYTORCH_OP_CONVERSION_CHECK(body_parameters.size() > 0, "At least one input to Loop body is required");
    // Set counter shape
    body_parameters[0]->set_partial_shape(PartialShape{});
    // #0 loop input is  trip_count, #1 loop input is condition
    // Connect other inputs
    for (size_t i = 2; i < inputs.size(); i++) {
        if (i <= subgraph_decoder->num_of_outputs()) {
            loop->set_merged_input(body_parameters[i - 1], inputs[i], body_results[i - 1]);
        } else {
            loop->set_invariant_input(body_parameters[i - 1], inputs[i]);
        }
    }
    // Connect inputs from external context
    for (auto i = inputs.size() - 1; i < body_parameters.size(); i++) {
        auto param = body_parameters[i];
        auto input_idx = session->decode_tensor_name(param->output(0));
        auto external_output = context.get_tensor_from_model_or_create_input(input_idx);
        if (output_idxs.count(input_idx)) {
            loop->set_merged_input(param, external_output, output_idxs.at(input_idx));
        } else {
            loop->set_invariant_input(param, external_output);
        }
    }

    // connect outputs
    for (size_t i = 1; i < body_results.size(); i++) {
        auto result = body_results[i];
        auto out_idx = session->decode_tensor_name(result->input(0).get_source_output());
        context.add_tensor_to_context(out_idx, loop->get_iter_value(result, -1));
    }
    loop->validate_and_infer_types();
    return {context.mark_node(loop)->outputs()};
};

OutputVector translate_while_loop_fx(const NodeContext& context) {
    // FX while_loop has structure:
    // - 2 subgraphs: cond_fn (index 0) and body_fn (index 1)
    // - inputs = carried_inputs (values that are updated each iteration)
    // - cond_fn takes carried_inputs, returns boolean scalar
    // - body_fn takes carried_inputs, returns new carried_inputs
    //
    // OpenVINO Loop requires:
    // - trip_count (max iterations)
    // - execution_condition (initial condition)
    // - body model with Parameters and Results where:
    //   - Parameter[0] is iteration counter
    //   - Result[0] is next condition
    //   - Other parameters/results are carried values

    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(
        decoder->get_subgraph_size() == 2,
        "while_loop must have 2 subgraphs (cond and body), got " + std::to_string(decoder->get_subgraph_size()));

    const auto& inputs = context.inputs();
    const size_t num_carried = inputs.size();
    PYTORCH_OP_CONVERSION_CHECK(num_carried > 0, "while_loop must have at least one carried input.");

    // Convert cond and body subgraphs
    auto cond_model = context.convert_subgraph(0);
    auto body_model = context.convert_subgraph(1);

    auto cond_params = cond_model->get_parameters();
    auto cond_results = cond_model->get_results();
    auto body_params = body_model->get_parameters();
    auto body_results = body_model->get_results();

    PYTORCH_OP_CONVERSION_CHECK(cond_results.size() == 1, "cond_fn must have exactly one output (boolean condition).");
    PYTORCH_OP_CONVERSION_CHECK(body_results.size() == num_carried,
                                "body_fn must return same number of outputs as carried inputs. Expected " +
                                    std::to_string(num_carried) + ", got " + std::to_string(body_results.size()));

    // Build combined body model for OpenVINO Loop:
    // Parameters: [counter, ...carried_inputs]
    // Results: [condition, ...new_carried_values]
    //
    // Implementation approach:
    // 1. Create parameters for [counter, carried_inputs]
    // 2. Clone body operations connected to new parameters
    // 3. Clone cond operations connected to body outputs
    // 4. Return [cond_output, body_outputs]

    // Create trip_count (large number for while loop)
    auto trip_count =
        ov::opset10::Constant::create(ov::element::i64, ov::Shape{}, {std::numeric_limits<int32_t>::max()});

    // Create initial condition (true)
    auto init_cond = ov::opset10::Constant::create(ov::element::boolean, ov::Shape{}, {true});

    // For FX while_loop, we need to construct a proper body
    // The body should be: [counter, x0, x1, ...] -> [cond(body(x0,x1,...)), body(x0,x1,...)]

    // Create new parameters for the loop body
    ov::ParameterVector loop_body_params;
    // counter
    auto loop_counter = std::make_shared<ov::opset10::Parameter>(ov::element::i64, ov::PartialShape{});
    loop_counter->set_friendly_name("loop_iteration");
    loop_body_params.push_back(std::move(loop_counter));

    // carried inputs
    for (size_t i = 0; i < num_carried; i++) {
        auto param =
            std::make_shared<ov::opset10::Parameter>(inputs[i].get_element_type(), inputs[i].get_partial_shape());
        param->set_friendly_name("loop_carried_" + std::to_string(i));
        loop_body_params.push_back(std::move(param));
    }

    // Map body model parameters to loop body parameters
    std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> body_param_map;
    for (size_t i = 0; i < body_params.size() && i < num_carried; i++) {
        body_param_map[body_params[i]] = loop_body_params[i + 1];
    }

    // Clone body model operations
    ov::NodeVector body_ops;
    std::map<ov::Node*, ov::Node*> node_map;
    for (const auto& orig_param : body_params) {
        if (body_param_map.count(orig_param)) {
            node_map[orig_param.get()] = body_param_map[orig_param].get();
        }
    }

    // Clone all body operations (except parameters and results)
    for (const auto& op : body_model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Parameter>(op) || ov::is_type<ov::opset10::Result>(op)) {
            continue;
        }
        ov::OutputVector new_inputs;
        for (const auto& input : op->inputs()) {
            auto source = input.get_source_output();
            auto source_node = source.get_node();
            if (node_map.count(source_node)) {
                new_inputs.push_back(node_map[source_node]->output(source.get_index()));
            } else {
                new_inputs.push_back(source);
            }
        }
        auto cloned = op->clone_with_new_inputs(new_inputs);
        cloned->set_friendly_name(op->get_friendly_name());
        node_map[op.get()] = cloned.get();
        body_ops.push_back(cloned);
    }

    // Get cloned body outputs
    std::vector<ov::Output<ov::Node>> cloned_body_outputs;
    for (const auto& result : body_results) {
        auto source = result->input(0).get_source_output();
        auto source_node = source.get_node();
        if (node_map.count(source_node)) {
            cloned_body_outputs.push_back(node_map[source_node]->output(source.get_index()));
        } else {
            cloned_body_outputs.push_back(source);
        }
    }

    // Now clone cond model and connect to cloned body outputs
    // Create separate maps for cond params -> body outputs and cloned operations
    std::map<ov::Node*, size_t> cond_param_to_idx;  // Maps cond param to index
    for (size_t i = 0; i < cond_params.size(); i++) {
        cond_param_to_idx[cond_params[i].get()] = i;
    }
    std::map<ov::Node*, ov::Node*> cond_cloned_map;  // Maps original cond ops to cloned ops

    // Clone cond operations
    for (const auto& op : cond_model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Parameter>(op) || ov::is_type<ov::opset10::Result>(op)) {
            continue;
        }
        ov::OutputVector new_inputs;
        for (const auto& input : op->inputs()) {
            auto source = input.get_source_output();
            auto source_node = source.get_node();
            if (cond_param_to_idx.count(source_node)) {
                // This input is from a cond parameter - connect to corresponding body output
                size_t param_idx = cond_param_to_idx.at(source_node);
                new_inputs.push_back(cloned_body_outputs[param_idx]);
            } else if (cond_cloned_map.count(source_node)) {
                // This input is from a previously cloned cond operation
                new_inputs.push_back(cond_cloned_map[source_node]->output(source.get_index()));
            } else if (node_map.count(source_node)) {
                // This input is from a body operation
                new_inputs.push_back(node_map[source_node]->output(source.get_index()));
            } else {
                // Keep original (e.g., constants from outside)
                new_inputs.push_back(source);
            }
        }
        auto cloned = op->clone_with_new_inputs(new_inputs);
        cloned->set_friendly_name(op->get_friendly_name() + "_cond");
        cond_cloned_map[op.get()] = cloned.get();
        body_ops.push_back(std::move(cloned));
    }

    // Get condition output
    ov::Output<ov::Node> cond_output;
    for (const auto& result : cond_results) {
        auto source = result->input(0).get_source_output();
        if (cond_cloned_map.count(source.get_node())) {
            cond_output = cond_cloned_map[source.get_node()]->output(source.get_index());
        } else {
            cond_output = source;
        }
    }

    // Ensure condition is scalar boolean
    if (cond_output.get_partial_shape().rank().is_static() && cond_output.get_partial_shape().rank().get_length() > 0) {
        cond_output = std::make_shared<ov::opset10::Squeeze>(cond_output);
    }

    // Create results: [condition, body_output_0, body_output_1, ...]
    ov::ResultVector loop_body_results;
    loop_body_results.push_back(std::make_shared<ov::opset10::Result>(cond_output));
    for (const auto& out : cloned_body_outputs) {
        loop_body_results.push_back(std::make_shared<ov::opset10::Result>(out));
    }

    // Create body model
    auto loop_body = std::make_shared<ov::Model>(loop_body_results, loop_body_params, "while_loop_body");

    // Create Loop operation
    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, init_cond);
    loop->set_function(loop_body);

    // Set special body ports: counter at index 0, condition output at index 0
    ov::op::v5::Loop::SpecialBodyPorts spec_ports{0, 0};
    loop->set_special_body_ports(spec_ports);

    // Set merged inputs (carried values)
    for (size_t i = 0; i < num_carried; i++) {
        loop->set_merged_input(loop_body_params[i + 1], inputs[i], loop_body_results[i + 1]);
    }

    // Validate and get outputs
    loop->validate_and_infer_types();

    // Get outputs (skip condition result which is index 0)
    OutputVector outputs;
    for (size_t i = 0; i < num_carried; i++) {
        auto out = loop->get_iter_value(loop_body_results[i + 1], -1);
        outputs.push_back(std::move(out));
    }

    context.mark_node(loop);

    // In FX, while_loop returns a tuple that is accessed via getitem.
    // We wrap outputs in make_list_construct so getitem can extract individual elements.
    return {context.mark_node(make_list_construct(outputs))};
}

OutputVector translate_scan_fx(const NodeContext& context) {
    // torch.ops.higher_order.scan(combine_fn, init, xs, additional_inputs) calls
    //     combine_fn(*carry, *x_slices, *additional_inputs) -> (*next_carry, *ys)
    // once per slice of xs along dim 0, starting from carry = init, and returns (*final_carry, *stacked_ys).
    //
    // This maps onto ov::op::v5::Loop with the converted combine_fn as its body:
    // - init              -> merged inputs (carried from one iteration to the next)
    // - xs                -> sliced inputs along axis 0 (the body sees one slice, without axis 0)
    // - additional_inputs -> invariant inputs
    // - ys                -> concatenated outputs along axis 0
    // The trip count is xs[0].shape[0], computed at runtime so a dynamic sequence length works.
    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(
        decoder->get_subgraph_size() == 1,
        "scan must have 1 subgraph (combine_fn), got " + std::to_string(decoder->get_subgraph_size()));
    // The decoder flattens init, xs and additional_inputs into inputs(): it reports how many inputs each one became.
    PYTORCH_OP_CONVERSION_CHECK(context.has_attribute("__openvino_hop_input_group_sizes"),
                                "scan requires the decoder to report the sizes of init, xs and additional_inputs.");
    const auto group_sizes = context.get_attribute<std::vector<int64_t>>("__openvino_hop_input_group_sizes");
    PYTORCH_OP_CONVERSION_CHECK(
        group_sizes.size() == 3,
        "scan expects 3 input groups (init, xs, additional_inputs), got " + std::to_string(group_sizes.size()));
    const auto num_init = static_cast<size_t>(group_sizes[0]);
    const auto num_xs = static_cast<size_t>(group_sizes[1]);
    const auto num_additional = static_cast<size_t>(group_sizes[2]);

    const auto& inputs = context.inputs();
    PYTORCH_OP_CONVERSION_CHECK(num_init + num_xs + num_additional == inputs.size(),
                                "scan input group sizes do not add up to its number of inputs.");
    PYTORCH_OP_CONVERSION_CHECK(num_xs > 0, "scan must have at least one xs to iterate over.");

    auto body = context.convert_subgraph(0);
    const auto body_params = body->get_parameters();
    const auto body_results = body->get_results();
    PYTORCH_OP_CONVERSION_CHECK(body_params.size() == inputs.size(),
                                "scan combine_fn must take one parameter per input. Expected " +
                                    std::to_string(inputs.size()) + ", got " + std::to_string(body_params.size()));
    PYTORCH_OP_CONVERSION_CHECK(body_results.size() >= num_init,
                                "scan combine_fn must return at least the next carry values.");

    // Loop slices xs with part_size 1, so the body receives [1, ...]: squeeze axis 0 to give combine_fn the slice it
    // expects. Each xs parameter is replaced with a new one that feeds that Squeeze.
    auto axis_0 = v0::Constant::create(element::i64, Shape{1}, {0});
    std::vector<std::shared_ptr<v0::Parameter>> sliced_params;
    for (size_t i = num_init; i < num_init + num_xs; i++) {
        const auto& param = body_params[i];
        auto sliced_param = std::make_shared<v0::Parameter>(param->get_element_type(), PartialShape::dynamic());
        sliced_param->set_friendly_name(param->get_friendly_name());
        param->output(0).replace(std::make_shared<v0::Squeeze>(sliced_param, axis_0));
        body->replace_parameter(body->get_parameter_index(param), sliced_param);
        sliced_params.push_back(std::move(sliced_param));
    }

    // Each ys is one step's slice: unsqueeze axis 0 so Loop concatenates the steps into [seq, ...].
    for (size_t i = num_init; i < body_results.size(); i++) {
        const auto& result = body_results[i];
        result->input(0).replace_source_output(std::make_shared<v0::Unsqueeze>(result->input_value(0), axis_0));
    }

    // scan always runs every step, so the body's condition is a constant true and there is no iteration counter.
    auto body_condition = std::make_shared<v0::Result>(v0::Constant::create(element::boolean, Shape{}, {true}));
    body->add_results({body_condition});

    auto xs_shape = context.mark_node(std::make_shared<v3::ShapeOf>(inputs[num_init], element::i64));
    auto trip_count = context.mark_node(std::make_shared<v8::Gather>(xs_shape,
                                                                     v0::Constant::create(element::i64, Shape{}, {0}),
                                                                     v0::Constant::create(element::i64, Shape{}, {0})));
    auto execution_condition = v0::Constant::create(element::boolean, Shape{}, {true});

    auto loop = std::make_shared<v5::Loop>(trip_count, execution_condition);
    loop->set_function(body);
    loop->set_special_body_ports({-1, static_cast<int64_t>(body->get_result_index(body_condition))});

    for (size_t i = 0; i < num_init; i++) {
        loop->set_merged_input(body_params[i], inputs[i], body_results[i]);
    }
    for (size_t i = 0; i < num_xs; i++) {
        loop->set_sliced_input(sliced_params[i], inputs[num_init + i], 0, 1, 1, -1, 0);
    }
    for (size_t i = num_init + num_xs; i < inputs.size(); i++) {
        loop->set_invariant_input(body_params[i], inputs[i]);
    }

    OutputVector outputs;
    for (size_t i = 0; i < num_init; i++) {
        outputs.push_back(loop->get_iter_value(body_results[i], -1));
    }
    for (size_t i = num_init; i < body_results.size(); i++) {
        outputs.push_back(loop->get_concatenated_slices(body_results[i], 0, 1, 1, -1, 0));
    }
    loop->validate_and_infer_types();
    context.mark_node(loop);

    // Like while_loop, scan returns a tuple that FX unpacks with getitem.
    return {context.mark_node(make_list_construct(outputs))};
}

}  // namespace ov::frontend::pytorch::op
