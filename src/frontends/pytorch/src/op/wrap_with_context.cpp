// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_wrap_with_context_fx(const NodeContext& context) {
    // wrap_with_set_grad_enabled(enabled, body, *operands) and
    // wrap_with_autocast(device_type, dtype, enabled, cache_enabled, body, *operands) run the body under a
    // context manager. Gradient mode and autocast do not change inference semantics, so the body is inlined.
    // Decoder inputs are the inlined context arguments followed by the operands.
    const auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 1,
                                context.get_op_type(),
                                " must have exactly 1 subgraph, got: ",
                                decoder->get_subgraph_size());
    const auto body_decoder = decoder->get_subgraph_decoder(0);
    const auto body_inputs = body_decoder->inputs();
    PYTORCH_OP_CONVERSION_CHECK(context.get_input_size() >= body_inputs.size(),
                                context.get_op_type(),
                                " body expects ",
                                body_inputs.size(),
                                " operands, got: ",
                                context.get_input_size());
    const auto first_operand = context.get_input_size() - body_inputs.size();

    const auto body = context.convert_subgraph(0);
    const auto session = context.get_session();
    const auto get_operand = [&](size_t body_tensor) {
        const auto operand = std::find(body_inputs.begin(), body_inputs.end(), body_tensor);
        PYTORCH_OP_CONVERSION_CHECK(operand != body_inputs.end(),
                                    context.get_op_type(),
                                    " body references a value which is not its operand.");
        return first_operand + static_cast<size_t>(operand - body_inputs.begin());
    };

    const auto num_outputs = body_decoder->num_of_outputs();
    const auto& results = body->get_results();
    PYTORCH_OP_CONVERSION_CHECK(results.size() >= num_outputs,
                                context.get_op_type(),
                                " body has fewer outputs than expected.");
    // Results after the declared outputs hold operands mutated by the body.
    std::vector<std::pair<size_t, Output<Node>>> mutations;
    for (size_t i = num_outputs; i < results.size(); ++i) {
        const auto value = results[i]->input_value(0);
        mutations.emplace_back(get_operand(session->decode_tensor_name(value)), value);
    }
    std::vector<std::pair<std::shared_ptr<v0::Parameter>, size_t>> parameters;
    for (const auto& parameter : body->get_parameters()) {
        parameters.emplace_back(parameter, get_operand(session->decode_tensor_name(parameter->output(0))));
    }

    // Positions of outputs which are views of operands are computed from body parameters, so they must be created
    // before the parameters are replaced.
    const auto aliases = session->take_subgraph_output_aliases(body);
    for (const auto& alias : aliases) {
        alias.positions->get();
    }
    const auto body_nodes = body->get_ordered_ops();
    // Body tensor names index the body graph and must not be decoded as parent tensors.
    for (const auto& node : body_nodes) {
        for (auto& output : node->outputs()) {
            output.set_names({});
        }
    }
    for (const auto& [parameter, operand] : parameters) {
        parameter->output(0).replace(context.get_input(static_cast<int>(operand)));
    }
    for (const auto& node : body_nodes) {
        node->revalidate_and_infer_types();
    }

    OutputVector outputs;
    for (size_t i = 0; i < num_outputs; ++i) {
        outputs.push_back(results[i]->input_value(0));
    }
    for (const auto& [operand, value] : mutations) {
        context.mutate_input(operand, value);
    }
    const auto& operand_ids = decoder->inputs();
    for (const auto& alias : aliases) {
        const auto operand = get_operand(alias.root_id);
        if (operand_ids.at(operand) == 0 && decoder->is_input_inlined(operand)) {
            continue;
        }
        session->m_tuple_element_aliases[outputs.at(alias.output_index)] = {
            operand_ids.at(operand),
            decoder,
            outputs.at(alias.output_index),
            context.get_input(static_cast<int>(operand)),
            {},
            alias.positions};
    }
    // The wrapper returns a tuple even for a single value; parent getitem nodes select its elements.
    return {make_list_construct(outputs)};
}

}  // namespace ov::frontend::pytorch::op
