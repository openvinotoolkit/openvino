// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_call_method(const NodeContext& context) {
    // prim::CallMethod[name="forward"](module, args...) -> outputs
    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() > 0,
                                "prim::CallMethod must have at least one subgraph (the method body).");

    auto body_model = context.convert_subgraph(0);
    auto parameters = body_model->get_parameters();
    auto inputs = context.inputs();

    PYTORCH_OP_CONVERSION_CHECK(parameters.size() <= inputs.size(),
                                "Method body has more parameters than provided inputs. Got: ",
                                inputs.size(),
                                " inputs, Expected: ",
                                parameters.size(),
                                " parameters.");

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
