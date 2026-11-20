// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_trunc(const NodeContext& context) {
    // aten::trunc(Tensor self) -> Tensor
    // aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 1, 2);
    return {trunc_toward_zero(context, context.get_input(0))};
};

OutputVector translate_trunc_to_int(const NodeContext& context) {
    // math.trunc(SymFloat) -> SymInt
    num_inputs_check(context, 1, 1);
    auto truncated = trunc_toward_zero(context, context.get_input(0));
    return {context.mark_node(std::make_shared<ov::op::v0::Convert>(truncated, element::i64))};
};

}  // namespace ov::frontend::pytorch::op
