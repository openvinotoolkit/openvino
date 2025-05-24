// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_pop(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    auto list_elems = get_list_as_outputs(context.get_input(0));
    if (list_elems.empty())
        throw std::runtime_error("pop from empty list");
    size_t list_size = list_elems.size();

    int64_t pop_index = -1;
    if (!context.input_is_none(1)) {
        auto node = context.get_input(1);
        auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node.get_node_shared_ptr());
        if (!constant)
            throw std::runtime_error("pop index must be a constant integer");
        auto values = constant->cast_vector<int64_t>();
        if (values.empty())
            throw std::runtime_error("pop index constant is empty");
        pop_index = values[0];
    }

    if (pop_index == -1) {
        pop_index = list_size - 1;
    } else if (pop_index < 0) {
        pop_index += list_size;
    }
    if (pop_index < 0 || pop_index >= static_cast<int64_t>(list_size))
        throw std::runtime_error("pop index out of range");

    auto result = list_elems[pop_index];
    list_elems.erase(list_elems.begin() + pop_index);

    if (!context.input_is_none(1)) {
        context.mutate_input(1, result);
    }

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
