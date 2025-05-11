// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_int(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    return {context.mark_node(std::make_shared<ov::op::v0::Convert>(context.get_input(0), element::i64))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
