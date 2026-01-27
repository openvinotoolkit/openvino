// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/loop.hpp"

#include <utility>

#include "core/graph.hpp"
#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/identity.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "translate_session.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
namespace {
/// \brief      Check if termination condition is true during all Loop
///             iterations.
///             It allows to replace termination condition body output with
///             Constant.
///             As a result OV Loop shape inference is able to handle more
///             cases.
///
/// \param[in]  cond_in    boolean input to the loop body depicting loop termination condition
///
/// \param[in]  cond_out   loop termination condition computed after each iteration
///
/// \return true if termination condition is not modified during loop iterations, false otherwise.
bool is_termination_condition_always_true(const ov::Node* cond_in, const ov::Node* cond_out) {
    if (auto identity = ov::as_type<const op::v16::Identity>(cond_out)) {
        cond_out = identity->input_value(0).get_node();
    }
    return cond_in == cond_out;
}

using ParameterPtr = std::shared_ptr<ov::op::v0::Parameter>;
using InvariantInput = std::pair<ParameterPtr, ov::Output<ov::Node>>;
/// \brief Splits loop body parameters into canonical (iteration/condition/state) and invariants.
///
/// Parameters that match tensors already defined in the parent translate session are classified as
/// invariant inputs so that they can be wired via Loop::set_invariant_input, while the rest remain
/// canonical parameters that participate in Loop body state.
std::pair<ov::ParameterVector, std::vector<InvariantInput>> partition_body_parameters(
    const ov::frontend::onnx::Node& node,
    const ov::ParameterVector& params) {
    ov::ParameterVector canonical;
    canonical.reserve(params.size());
    std::vector<InvariantInput> invariants;
    invariants.reserve(params.size());

    const auto translate_session = node.get_translate_session();
    FRONT_END_OP_CONVERSION_CHECK(translate_session != nullptr,
                                  "Translate session is required to partition Loop body parameters");
    for (const auto& param : params) {
        ov::Output<ov::Node> known_input;
        const auto& names = param->output(0).get_names();
        for (const auto& name : names) {
            known_input = translate_session->lookup_tensor(name);
            if (known_input.get_node() != nullptr) {
                break;
            }
        }
        if (known_input.get_node() != nullptr) {
            invariants.emplace_back(param, known_input);
        } else {
            canonical.push_back(param);
        }
    }

    return {std::move(canonical), std::move(invariants)};
}
}  // namespace

namespace detail {
ov::OutputVector loop_legacy(const ov::frontend::onnx::Node& node) {
    const auto& ng_inputs = node.get_ov_inputs();

    const ov::OutputVector loop_carried_dependencies{std::next(ng_inputs.begin(), 2), ng_inputs.end()};

    const auto& subgraphs = node.get_subgraphs();
    auto body_graph = subgraphs.at("body");
    auto body_outputs = body_graph->get_ov_outputs();
    const auto& body_inputs = body_graph->get_ng_parameters();

    // Infer loop body inputs' element type based on carried dependencies
    for (size_t i = 0; i < loop_carried_dependencies.size(); i++) {
        body_inputs[i + 2]->set_element_type(loop_carried_dependencies[i].get_element_type());
        body_inputs[i + 2]->set_partial_shape(loop_carried_dependencies[i].get_partial_shape());
    }

    // optional inputs
    ov::Output<ov::Node> trip_count;
    // trip count skipped or has value max(int64_t) means infinitive loop
    if (ov::op::util::is_null(ng_inputs.at(0)) ||
        (ov::op::util::is_constant(ng_inputs.at(0).get_node_shared_ptr()) &&
         ov::as_type_ptr<v0::Constant>(ng_inputs.at(0).get_node_shared_ptr())->cast_vector<int64_t>()[0] ==
             std::numeric_limits<int64_t>::max())) {
        // -1 means infinite Loop
        trip_count = v0::Constant::create(ov::element::i64, {1}, {-1});
    } else {
        trip_count = ng_inputs.at(0);
    }

    ov::Output<ov::Node> termination_cond;                             // true means that first interation should be run
    if (ov::op::util::is_null(ng_inputs.at(1).get_node_shared_ptr()))  // termination condition skipped
    {
        termination_cond = v0::Constant::create(ov::element::boolean, {1}, {true});
    } else if (ov::op::util::is_constant(ng_inputs.at(1).get_node_shared_ptr()) &&
               ov::as_type_ptr<v0::Constant>(ng_inputs.at(1).get_node_shared_ptr())->cast_vector<bool>()[0] == false) {
        // no iteration is performed so initial values are returned
        ov::OutputVector node_outputs;
        // final values
        for (const auto& dep : loop_carried_dependencies) {
            node_outputs.push_back(dep);
        }
        // scan outputs
        for (const auto& dep : loop_carried_dependencies) {
            node_outputs.push_back(dep);
        }
        return node_outputs;
    } else {
        termination_cond = ng_inputs.at(1);
    }

    const int64_t concat_axis = 0;
    const auto concat_axis_const = v0::Constant::create(ov::element::i64, {1}, {concat_axis});
    // add dimension along which scan outputs will be concatenated
    for (size_t i = loop_carried_dependencies.size() + 1; i < body_outputs.size(); ++i) {
        body_outputs[i] = std::make_shared<v0::Unsqueeze>(body_outputs[i], concat_axis_const);
    }

    const auto& cond_in = body_inputs[1];
    const auto& cond_out = body_outputs[0];
    bool needs_condition_param = true;
    ov::ParameterVector body_params;
    // optimization allow to improve nG Loop shape inference
    if (is_termination_condition_always_true(cond_in.get(), cond_out.get_node())) {
        body_outputs[0] = v0::Constant::create(ov::element::boolean, {1}, {true});
        // Construct body_params without the condition parameter (body_inputs[1])
        body_params = ov::ParameterVector{body_inputs[0]};
        body_params.insert(body_params.end(), body_inputs.begin() + 2, body_inputs.end());
        needs_condition_param = false;
    } else {
        // Construct body_params with all body_inputs
        body_params = ov::ParameterVector(body_inputs.begin(), body_inputs.end());
    }

    CHECK_VALID_NODE(node,
                     body_inputs.size() >= loop_carried_dependencies.size() + 2,
                     "The provided loop body graph inputs size (",
                     body_inputs.size(),
                     "), is not greater than the sum of loop carried dependencies "
                     "and two mandatory"
                     " inputs (",
                     loop_carried_dependencies.size() + 2,
                     ")");

    CHECK_VALID_NODE(node,
                     body_outputs.size() >= loop_carried_dependencies.size() + 1,
                     "The provided loop body graph outputs size (",
                     body_outputs.size(),
                     ") is not greater than number of outputs. Required at least: ",
                     loop_carried_dependencies.size() + 1);

    const auto body = std::make_shared<ov::Model>(body_outputs, body_params);
    auto loop = std::make_shared<v5::Loop>(trip_count, termination_cond);
    v5::Loop::SpecialBodyPorts spec_ports{0, 0};
    loop->set_special_body_ports(spec_ports);
    loop->set_function(body);
    // Add condition
    if (needs_condition_param) {
        loop->set_merged_input(*std::next(body_inputs.begin(), 1), ng_inputs.at(1), *body_outputs.begin());
    }
    // Setting up other Loop body inputs.
    // body_inputs[0] is iteration number, body_inputs[1] is termination condition
    auto body_inputs_it = std::next(body_inputs.begin(), 2);
    // body_outputs[0] is termination condition output
    auto body_outputs_it = std::next(body_outputs.begin(), 1);

    // Set-up loop carried dependencies and final output values
    ov::OutputVector final_values;
    for (const auto& dep : loop_carried_dependencies) {
        loop->set_merged_input(*body_inputs_it++, dep, *body_outputs_it);
        final_values.push_back(loop->get_iter_value(*body_outputs_it++, -1));
    }

    const auto& inputs_from_parent = body_graph->get_inputs_from_parent();
    CHECK_VALID_NODE(node,
                     static_cast<size_t>(std::distance(body_inputs_it, body_inputs.end())) == inputs_from_parent.size(),
                     "Expected number of invariant parameters is"
                     " not equal number of provided inputs from parent scope");

    // Set-up parameters from parent graph which are not changed during Loop's
    // iterations
    for (auto in_from_parent_it = inputs_from_parent.begin();
         body_inputs_it != body_inputs.end() && in_from_parent_it != inputs_from_parent.end();
         ++body_inputs_it, ++in_from_parent_it) {
        loop->set_invariant_input(*body_inputs_it, *in_from_parent_it);
    }

    // Set-up scan outputs
    ov::OutputVector scan_outputs;
    for (; body_outputs_it != body_outputs.end(); body_outputs_it++) {
        // start=0, stride=1, part_size=1, end=-1, axis=0
        scan_outputs.push_back(loop->get_concatenated_slices(*body_outputs_it, 0, 1, 1, -1, concat_axis));
    }
    loop->validate_and_infer_types();

    ov::OutputVector node_outputs;
    for (const auto& v : final_values) {
        node_outputs.push_back(v);
    }
    for (const auto& v : scan_outputs) {
        node_outputs.push_back(v);
    }
    return node_outputs;
}

ov::OutputVector loop(const ov::frontend::onnx::Node& node) {
    const auto& ng_inputs = node.get_ov_inputs();

    const ov::OutputVector loop_carried_dependencies{std::next(ng_inputs.begin(), 2), ng_inputs.end()};

    auto body_graph = node.get_attribute_value<std::shared_ptr<ov::Model>>("body");
    const auto& body_results = body_graph->get_results();
    ov::OutputVector body_outputs;
    body_outputs.reserve(body_results.size());
    for (const auto& res : body_results) {
        body_outputs.push_back(res->get_input_source_output(0));
    }
    auto body_inputs = body_graph->get_parameters();

    auto [canonical_inputs, invariant_inputs] = partition_body_parameters(node, body_inputs);

    CHECK_VALID_NODE(node,
                     canonical_inputs.size() >= 2,
                     "The provided loop body graph inputs size (",
                     canonical_inputs.size(),
                     ") is not greater than the mandatory iteration and condition inputs (2)");

    auto iteration_param = canonical_inputs[0];
    auto condition_param = canonical_inputs[1];
    ov::ParameterVector state_parameters(canonical_inputs.begin() + 2, canonical_inputs.end());

    const auto default_trip_count = v0::Constant::create(ov::element::i64, {1}, {-1});
    const auto true_condition = v0::Constant::create(ov::element::boolean, {1}, {true});

    // optional inputs
    ov::Output<ov::Node> trip_count;
    const auto& max_iterations_input = ng_inputs.at(0);
    const bool infinite_trip_count =
        ov::op::util::is_null(max_iterations_input) ||
        (ov::op::util::is_constant(max_iterations_input.get_node_shared_ptr()) &&
         ov::as_type_ptr<v0::Constant>(max_iterations_input.get_node_shared_ptr())->cast_vector<int64_t>()[0] ==
             std::numeric_limits<int64_t>::max());
    trip_count = infinite_trip_count ? default_trip_count : max_iterations_input;

    ov::Output<ov::Node> termination_cond;  // true means that first iteration should be run
    const auto& cond_input = ng_inputs.at(1);
    if (ov::op::util::is_null(cond_input.get_node_shared_ptr())) {
        termination_cond = true_condition;
    } else if (ov::op::util::is_constant(cond_input.get_node_shared_ptr()) &&
               !ov::as_type_ptr<v0::Constant>(cond_input.get_node_shared_ptr())->cast_vector<bool>()[0]) {
        ov::OutputVector node_outputs;
        node_outputs.reserve(loop_carried_dependencies.size() * 2);
        for (const auto& dep : loop_carried_dependencies) {
            node_outputs.push_back(dep);
        }
        for (const auto& dep : loop_carried_dependencies) {
            node_outputs.push_back(dep);
        }
        return node_outputs;
    } else {
        termination_cond = cond_input;
    }

    const int64_t concat_axis = 0;
    const auto concat_axis_const = v0::Constant::create(ov::element::i64, {1}, {concat_axis});
    // add dimension along which scan outputs will be concatenated
    for (size_t i = loop_carried_dependencies.size() + 1; i < body_outputs.size(); ++i) {
        body_outputs[i] = std::make_shared<v0::Unsqueeze>(body_outputs[i], concat_axis_const);
    }

    bool needs_condition_param = true;
    if (body_outputs.empty()) {
        body_outputs.push_back(true_condition);
        needs_condition_param = false;
    } else {
        const auto& cond_out = body_outputs[0];
        // optimization allow to improve nG Loop shape inference
        if (is_termination_condition_always_true(condition_param.get(), cond_out.get_node())) {
            body_outputs[0] = true_condition;
            needs_condition_param = false;
        }
    }

    CHECK_VALID_NODE(node,
                     state_parameters.size() == loop_carried_dependencies.size(),
                     "Number of loop body state parameters (",
                     state_parameters.size(),
                     ") does not match number of loop carried dependencies (",
                     loop_carried_dependencies.size(),
                     ")");

    // Infer loop body inputs' element type based on carried dependencies
    for (size_t i = 0; i < loop_carried_dependencies.size(); i++) {
        state_parameters[i]->set_element_type(loop_carried_dependencies[i].get_element_type());
        state_parameters[i]->set_partial_shape(loop_carried_dependencies[i].get_partial_shape());
    }

    CHECK_VALID_NODE(node,
                     body_outputs.size() >= loop_carried_dependencies.size() + 1,
                     "The provided loop body graph outputs size (",
                     body_outputs.size(),
                     ") is not greater than number of outputs. Required at least: ",
                     loop_carried_dependencies.size() + 1);

    ov::ParameterVector body_params;
    body_params.push_back(iteration_param);
    if (needs_condition_param) {
        body_params.push_back(condition_param);
    }
    body_params.insert(body_params.end(), state_parameters.begin(), state_parameters.end());
    for (const auto& invariant : invariant_inputs) {
        body_params.push_back(invariant.first);
    }
    const auto body = std::make_shared<ov::Model>(body_outputs, body_params);
    auto loop = std::make_shared<v5::Loop>(trip_count, termination_cond);
    v5::Loop::SpecialBodyPorts spec_ports{0, 0};
    loop->set_special_body_ports(spec_ports);
    loop->set_function(body);
    if (needs_condition_param) {
        loop->set_merged_input(condition_param, termination_cond, body_outputs[0]);
    }

    // Setting up other Loop body inputs.
    auto body_inputs_it = state_parameters.begin();
    // body_outputs[0] is termination condition output
    auto body_outputs_it = std::next(body_outputs.begin(), 1);

    // Set-up loop carried dependencies and final output values
    ov::OutputVector final_values;
    for (const auto& dep : loop_carried_dependencies) {
        loop->set_merged_input(*body_inputs_it++, dep, *body_outputs_it);
        final_values.push_back(loop->get_iter_value(*body_outputs_it++, -1));
    }

    for (const auto& [param, value] : invariant_inputs) {
        FRONT_END_GENERAL_CHECK(value.get_node() != nullptr,
                                "Non-existent connection in body-graph to " + param->get_friendly_name());
        loop->set_invariant_input(param, value);
    }

    // Set-up scan outputs
    auto node_outputs = std::move(final_values);
    for (; body_outputs_it != body_outputs.end(); ++body_outputs_it) {
        node_outputs.push_back(loop->get_concatenated_slices(*body_outputs_it, 0, 1, 1, -1, concat_axis));
    }
    // Run shape inference for body
    loop->validate_and_infer_types();
    return node_outputs;
}
}  // namespace detail

ov::OutputVector loop(const ov::frontend::onnx::Node& node) {
    if (!node.has_decoder()) {
        return detail::loop_legacy(node);
    } else {
        return detail::loop(node);
    }
}
ONNX_OP("Loop", OPSET_SINCE(1), ai_onnx::opset_1::loop);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
