// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_contains(const NodeContext& context) {
    // Tensor membership: item in container -> scalar bool
    num_inputs_check(context, 2, 2);

    auto container = context.get_input(0);
    auto item = context.get_input(1);

    // Require compatible element types to avoid silent semantic changes from casting
    // PyTorch uses type promotion, but casting item to container dtype can change membership results
    auto container_et = container.get_element_type();
    auto item_et = item.get_element_type();
    PYTORCH_OP_CONVERSION_CHECK(
        container_et.is_dynamic() || item_et.is_dynamic() || container_et == item_et,
        "aten::__contains__: container and item must have matching element types.");

    // Compare item with all elements (broadcasts)
    auto equal_mask = context.mark_node(std::make_shared<v1::Equal>(container, item));

    // Reduce over all axes -> scalar bool
    auto axes = get_axes_range(context, 0);
    auto result = context.mark_node(std::make_shared<v1::ReduceLogicalOr>(equal_mask, axes, false));

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
