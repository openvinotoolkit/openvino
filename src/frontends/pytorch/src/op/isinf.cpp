// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/is_inf.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_isinf_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    return {context.mark_node(std::make_shared<ov::op::v10::IsInf>(input))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
