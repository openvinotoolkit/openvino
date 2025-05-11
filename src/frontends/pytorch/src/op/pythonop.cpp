// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_pythonop(const NodeContext& context) {
    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 1,
                                "PythonOp must have 1 subgraph to be able to translate it to OV.");
    auto body = context.convert_subgraph(0);
    auto session = context.get_session();

    std::map<size_t, ParameterVector> inputs_map;
    for (const auto& param : body->get_parameters()) {
        auto tensor_idx = session->decode_tensor_name(param->output(0));
        PYTORCH_OP_CONVERSION_CHECK(!inputs_map.count(tensor_idx), "Multiple nodes with the same id are not allowed.");
        inputs_map[tensor_idx] = {param};
    }
    for (const auto& input : inputs_map) {
        auto external_output = context.get_input((int)input.first);
        if (external_output.get_node()) {
            input.second[0]->output(0).replace(external_output);
        }
    }

    OutputVector outputs{};
    for (auto& result : body->get_results()) {
        auto output = result->get_input_source_output(0);
        outputs.push_back(context.mark_output(output));
    }
    return outputs;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
