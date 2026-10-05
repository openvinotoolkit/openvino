// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/einsum.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_einsum(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    const auto equation = context.const_input<std::string>(0);
    const auto tensors = get_list_as_outputs(context.get_input(1));
    const OutputVector inputs(tensors.begin(), tensors.end());
    return {context.mark_node(std::make_shared<ov::op::v7::Einsum>(inputs, equation))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
