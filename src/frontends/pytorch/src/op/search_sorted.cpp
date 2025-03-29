// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/search_sorted.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_search_sorted(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    Output<Node> sorted;
    Output<Node> values;
    std::tie(sorted, values) = get_inputs_with_promoted_types(context, 0, 1);
    const bool out_int32 = context.const_input<bool>(2);
    PYTORCH_OP_CONVERSION_CHECK(out_int32 == false, "aten::searchsorted(out_int32=true) unsupported");
    const bool right_mode = context.const_input<bool>(3);
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(4), "aten::searchsorted(side) unsupported");
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(5), "aten::searchsorted(out) unsupported");
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(6), "aten::searchsorted(sorter) unsupported");
    auto op = context.mark_node(std::make_shared<ov::op::v15::SearchSorted>(sorted, values, right_mode));
    return {op};
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov