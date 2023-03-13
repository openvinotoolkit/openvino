// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_getitem(NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    if (std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(input.get_node_shared_ptr())) {
        FRONT_END_OP_CONVERSION_CHECK(!cast_fw_node(input.get_node_shared_ptr(), "aten::split"),
                                      "special case for aten::__getitem__");
        const auto&& list_elems = get_list_as_outputs(input);
        auto getitem_idx = context.const_input<int64_t>(1);
        if (getitem_idx < 0) {
            getitem_idx += list_elems.size();
        }
        FRONT_END_OP_CONVERSION_CHECK(getitem_idx < static_cast<int64_t>(list_elems.size()),
                                      "Index: ",
                                      getitem_idx,
                                      " is out of bounds of input list of len: ",
                                      list_elems.size());
        return {list_elems.at(getitem_idx)};
    }
    auto getitem_idx = context.get_input(1);
    auto zero = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<ov::op::v8::Gather>(input, getitem_idx, zero))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov