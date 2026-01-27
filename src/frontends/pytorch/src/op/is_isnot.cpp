// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_is(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    const bool lhs_is_none = context.input_is_none(0);
    const bool rhs_is_none = context.input_is_none(1);

    // Both are None - identity holds
    if (lhs_is_none && rhs_is_none) {
        return {context.mark_node(v0::Constant::create(element::boolean, {}, {true}))};
    }

    // One is None and other is not - identity fails
    if (lhs_is_none || rhs_is_none) {
        return {context.mark_node(v0::Constant::create(element::boolean, {}, {false}))};
    }

    // Tensor identity is not representable in OpenVINO IR
    PYTORCH_OP_CONVERSION_CHECK(false,
        "aten::__is__ supports only identity checks with None (x is None).");
    return {};
}

OutputVector translate_isnot(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    const bool lhs_is_none = context.input_is_none(0);
    const bool rhs_is_none = context.input_is_none(1);

    // Both are None - they are identical, so "is not" is false
    if (lhs_is_none && rhs_is_none) {
        return {context.mark_node(v0::Constant::create(element::boolean, {}, {false}))};
    }

    // One is None and other is not - they are not identical
    if (lhs_is_none || rhs_is_none) {
        return {context.mark_node(v0::Constant::create(element::boolean, {}, {true}))};
    }

    // Tensor identity is not representable in OpenVINO IR
    PYTORCH_OP_CONVERSION_CHECK(false,
        "aten::__isnot__ supports only identity checks with None (x is not None).");
    return {};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
