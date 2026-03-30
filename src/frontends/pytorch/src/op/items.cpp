// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_items(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto producer = input.get_node_shared_ptr();

    // Only support dict values created directly by prim::DictConstruct.
    if (auto dict_construct = cast_fw_node(producer, "prim::DictConstruct")) {
        const auto inputs = dict_construct->input_values();

        // DictConstruct inputs must be [key1, value1, key2, value2, ...].
        PYTORCH_OP_CONVERSION_CHECK(inputs.size() % 2 == 0,
                                    "aten::items: prim::DictConstruct inputs number is not divisible by 2.");

        OutputVector item_outputs;
        item_outputs.reserve(inputs.size() / 2);
        for (size_t i = 0; i < inputs.size(); i += 2) {
            auto key = inputs.at(i);
            auto value = inputs.at(i + 1);
            auto tuple = context.mark_node(make_list_construct({key, value}));
            item_outputs.push_back(tuple);
        }

        return {context.mark_node(make_list_construct(item_outputs))};
    }

    PYTORCH_OP_CONVERSION_CHECK(false, "aten::items: only Dict produced by prim::DictConstruct is supported.");
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
