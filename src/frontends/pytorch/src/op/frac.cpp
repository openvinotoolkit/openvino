// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_frac(const NodeContext& context) {
    // aten::frac(Tensor self) -> Tensor
    // aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    // frac(x) = x - trunc(x); the fractional part keeps the sign of x.
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    auto truncated = trunc_toward_zero(context, x);
    return {context.mark_node(std::make_shared<v1::Subtract>(x, truncated))};
}

}  // namespace ov::frontend::pytorch::op
