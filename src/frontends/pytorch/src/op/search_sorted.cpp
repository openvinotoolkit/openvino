// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/search_sorted.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_search_sorted(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    Output<Node> sorted;
    Output<Node> values;
    std::tie(sorted, values) = get_inputs_with_promoted_types(context, 0, 1);
    const bool out_int32 = get_const_input_or_attribute(context, 2, "out_int32", false);
    PYTORCH_OP_CONVERSION_CHECK(out_int32 == false, "aten::searchsorted(out_int32=true) unsupported");
    const bool right_mode = get_const_input_or_attribute(context, 3, "right", false);
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(4) && !context.has_attribute("side"),
                                "aten::searchsorted(side) unsupported");
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(5), "aten::searchsorted(out) unsupported");
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(6) && !context.has_attribute("sorter"),
                                "aten::searchsorted(sorter) unsupported");
    auto op = context.mark_node(std::make_shared<ov::op::v15::SearchSorted>(sorted, values, right_mode));
    return {op};
};
}  // namespace ov::frontend::pytorch::op