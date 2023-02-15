// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/util/op_types.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_pythonop(NodeContext& context) {
    auto decoder = context.get_decoder();
    FRONT_END_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 1,
                                  "PythonOp must have 1 subgraph to be able to translate it to OV.");
    auto body = context.convert_subgraph(0);

    std::map<int, ParameterVector> inputs_map;
    for (const auto& param : body->get_parameters()) {
        auto name = param->get_output_tensor(0).get_any_name();
        int input_idx = (int)std::stoll(name);
        if (inputs_map.count(input_idx)) {
            inputs_map[input_idx] = {param};
        } else {
            inputs_map[input_idx].push_back(param);
        }
    }
    for (const auto& input : inputs_map) {
        auto external_output = context.get_tensor_from_model(input.first);
        if (external_output.get_node()) {
            for (auto input_node : input.second) {
                replace_node(input_node, context.get_input(input.first).get_node_shared_ptr());
            }
        }
    }

    OutputVector outputs{};
    for (auto result : body->get_results()) {
        auto output = result->get_input_source_output(0).get_node_shared_ptr();
        context.mark_node(output);
        outputs.push_back(output);
    }
    return outputs;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
