// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_call_method(const NodeContext& context) {
    // prim::CallMethod[name="method_name"](module, inputs...)
    // The module is input 0.
    // The method to be called is usually available as a subgraph in the decoder.
    // We assume the first subgraph corresponds to the method being called.
    
    // Check if there are subgraphs available
    auto subgraph_size = context.get_decoder()->get_subgraph_size();
    FRONT_END_OP_CONVERSION_CHECK(subgraph_size > 0, "prim::CallMethod must have at least one subgraph (the method body).");

    // Convert the subgraph (method body)
    // The subgraph should handle the inputs (self + args) mapping internally via parameters.
    auto body_model = context.convert_subgraph(0);

    // Flatten the called graph into the current graph
    // We connect the inputs of CallMethod to the Parameters of the body_model
    // And return the Results of body_model as outputs.
    
    auto parameters = body_model->get_parameters();
    auto inputs = context.inputs();
    
    // There might be a mismatch if 'self' is implicit or explicit in specific ways,
    // but typically CallMethod inputs map 1:1 to the method graph parameters.
    FRONT_END_OP_CONVERSION_CHECK(parameters.size() <= inputs.size(), 
        "Method expects fewer or equal parameters than provided inputs. "
        "Got: ", inputs.size(), " inputs, Expected: ", parameters.size(), " parameters.");

    for (size_t i = 0; i < parameters.size(); ++i) {
        parameters[i]->output(0).replace(inputs[i]);
    }

    OutputVector outputs;
    for (const auto& result : body_model->get_results()) {
        outputs.push_back(result->input_value(0));
    }
    
    return outputs;
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
