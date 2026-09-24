// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_wrap_with_context_fx(const NodeContext& context) {
    // wrap_with_set_grad_enabled(enabled, body, *operands) and
    // wrap_with_autocast(device_type, dtype, enabled, cache_enabled, body, *operands) run the body under a
    // context manager. Gradient mode does not change inference semantics, so the body is inlined. Autocast changes
    // only the precision of operations; outputs are converted to the element types recorded by export.
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
    const auto aliases = session->take_subgraph_output_aliases(body);
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

    OutputVector outputs;
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output = results[i]->input_value(0);
        const auto type = simplified_type_interpret(body_decoder->get_output_type(i));
        if (type.is<element::Type>() && type.as<element::Type>().is_static() &&
            output.get_element_type() != type.as<element::Type>()) {
            output = context.mark_node(std::make_shared<v0::Convert>(output, type.as<element::Type>()));
        }
        outputs.push_back(output);
    }
    for (const auto& [operand, value] : mutations) {
        context.mutate_input(operand, value);
    }
    // Outputs which are views of operands are registered as aliases. The inlined view operations connect them to
    // the operand, so in-place updates are propagated like for any other view. The current operand value already
    // includes mutations made by the body.
    const auto& operand_ids = decoder->inputs();
    for (const auto& alias : aliases) {
        const auto operand = get_operand(alias.root_id);
        const auto& output = outputs.at(alias.output_index);
        session->m_tuple_element_aliases[{decoder->output(0), static_cast<int64_t>(alias.output_index)}] = {
            operand_ids.at(operand),
            decoder,
            output,
            context.get_input(static_cast<int>(operand))};
    }
    // The wrapper returns a tuple even for a single value; parent getitem nodes select its elements.
    return {make_list_construct(outputs)};
}

}  // namespace ov::frontend::pytorch::op
