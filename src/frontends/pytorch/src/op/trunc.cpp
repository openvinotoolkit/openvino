// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_trunc(const NodeContext& context) {
    // aten::trunc(Tensor self) -> Tensor
    // aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 1, 2);
    return {trunc_toward_zero(context, context.get_input(0))};
};

}  // namespace ov::frontend::pytorch::op
