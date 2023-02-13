// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_getitem(NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    FRONT_END_OP_CONVERSION_CHECK(cast_fw_node(input.get_node_shared_ptr(), "prim::ListConstruct") == nullptr,
                                  "unsupported case for aten::getitem");
    FRONT_END_OP_CONVERSION_CHECK(cast_fw_node(input.get_node_shared_ptr(), "aten::split") == nullptr,
                                  "unsupported case for aten::getitem");
    auto getitem_idx = context.get_input(1);
    auto zero = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<ov::op::v8::Gather>(input, getitem_idx, zero))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov