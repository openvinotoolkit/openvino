// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_wrap_with_autocast_fx(const NodeContext& context) {
    // wrap_with_autocast(device_type, dtype, enabled, cache_enabled, body, *operands)
    // Autocast changes only compute precision, so the body is inlined as is.
    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 1,
                                "wrap_with_autocast must have exactly 1 subgraph, got: ",
                                decoder->get_subgraph_size());
    constexpr size_t operands_start = 4;
    PYTORCH_OP_CONVERSION_CHECK(context.get_input_size() >= operands_start,
                                "wrap_with_autocast must have at least ",
                                operands_start,
                                " inputs.");
    const auto num_operands = context.get_input_size() - operands_start;
    auto body = context.convert_subgraph(0);
    auto session = context.get_session();

    const auto params = body->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        Output<Node> external;
        if (i < num_operands) {
            external = context.get_input(static_cast<int>(operands_start + i));
        } else {
            // Parameters beyond operands are tensors captured from the outer scope
            auto tensor_idx = session->decode_tensor_name(params[i]->output(0));
            external = context.get_tensor_from_model_or_create_input(tensor_idx);
        }
        params[i]->output(0).replace(external);
    }

    OutputVector outputs;
    for (const auto& result : body->get_results()) {
        outputs.push_back(context.mark_output(result->get_input_source_output(0)));
    }
    // The body returns a tuple, which is consumed by getitem
    return {context.mark_node(make_list_construct(outputs))};
}

}  // namespace ov::frontend::pytorch::op
