// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_prim_CallFunction(const NodeContext& context) {
    num_inputs_check(context, 2, context.get_input_size());

    auto decoder = context.get_decoder();
    
    // Check if the function is defined as a subgraph
    if (decoder->get_subgraph_size() > 0) {
        // Convert the subgraph (use first subgraph as the function body)
        auto body = context.convert_subgraph(0);

        // Map function arguments to subgraph parameters
        // In CallFunction, input 0 is the function reference, arguments start from input 1
        auto params = body->get_parameters();
        PYTORCH_OP_CONVERSION_CHECK(params.size() + 1 <= context.get_input_size(),
                                    "prim::CallFunction: Not enough arguments provided for subgraph parameters.");
        
        // Map parameters in order to function arguments (inputs 1, 2, 3, ...)
        for (size_t i = 0; i < params.size(); i++) {
            auto param = params[i];
            // Map parameter i to input i+1 (since input 0 is the function reference)
            auto external_output = context.get_input(static_cast<int>(i + 1));
            if (external_output.get_node()) {
                param->output(0).replace(external_output);
            }
        }

        // Return subgraph results
        OutputVector outputs;
        for (auto& result : body->get_results()) {
            auto output = result->get_input_source_output(0);
            outputs.push_back(context.mark_output(output));
        }

        // Handle output type conversion if needed
        if (outputs.size() > 0) {
            auto out_type = context.get_output_type(0);
            if (out_type.is<element::Type>()) {
                auto dtype = out_type.as<element::Type>();
                if (dtype.is_static() && dtype != outputs[0].get_element_type()) {
                    outputs[0] = context.mark_node(std::make_shared<v0::Convert>(outputs[0], dtype));
                }
            }
        }

        return outputs;
    }

    // Fallback: if no subgraph, return first argument (backward compatibility)
    PYTORCH_OP_CONVERSION_CHECK(context.get_input_size() > 1, 
                                "prim::CallFunction: No arguments provided");
    auto result = context.get_input(1);

    auto out_type = context.get_output_type(0);
    if (out_type.is<element::Type>()) {
        auto dtype = out_type.as<element::Type>();
        if (dtype.is_static() && dtype != result.get_element_type()) {
            result = context.mark_node(std::make_shared<v0::Convert>(result, dtype));
        }
    }

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
