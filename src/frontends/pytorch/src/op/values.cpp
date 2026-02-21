// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_values(const NodeContext& context) {
    // aten::values(Tensor self) -> Tensor
    num_inputs_check(context, 1, 1);
    return {context.get_input(0)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
