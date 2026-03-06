// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_prim_CallFunction(const NodeContext& context) {
    // prim::CallFunction can have only function reference input for zero-arg calls.
    num_inputs_check(context, 1, context.get_input_size());

    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 1,
                                "prim::CallFunction expects exactly one subgraph.");

    auto body = context.convert_subgraph(0);
    auto session = context.get_session();
    const auto& call_inputs = decoder->inputs();

    // In prim::CallFunction input 0 is function object, real arguments start from input 1.
    for (const auto& param : body->get_parameters()) {
        auto tensor_idx = session->decode_tensor_name(param->output(0));
        auto input_it = std::find(call_inputs.begin() + 1, call_inputs.end(), tensor_idx);
        Output<Node> external_output;
        if (input_it != call_inputs.end()) {
            auto input_pos = static_cast<int>(std::distance(call_inputs.begin(), input_it));
            external_output = context.get_input(input_pos);
        } else {
            external_output = context.get_tensor_from_model_or_create_input(tensor_idx);
        }
        if (external_output.get_node()) {
            param->output(0).replace(external_output);
        }
    }

    const auto& body_results = body->get_results();
    auto expected_outputs = context.get_output_size();
    PYTORCH_OP_CONVERSION_CHECK(body_results.size() >= expected_outputs,
                                "prim::CallFunction converted body has fewer outputs than expected: ",
                                body_results.size(),
                                " < ",
                                expected_outputs);

    OutputVector outputs;
    outputs.reserve(expected_outputs);
    for (size_t i = 0; i < expected_outputs; ++i) {
        outputs.push_back(context.mark_output(body_results[i]->input_value(0)));
    }

    // Additional body results can represent mutations propagated from nested scope.
    for (size_t i = expected_outputs; i < body_results.size(); ++i) {
        auto body_out = context.mark_output(body_results[i]->input_value(0));
        auto out_idx = session->decode_tensor_name(body_out);
        context.add_tensor_to_context(out_idx, body_out);
    }

    return outputs;
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
