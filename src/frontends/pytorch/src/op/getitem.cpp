// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_getitem(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    const auto idx_type = context.get_input_type(1);
    PYTORCH_OP_CONVERSION_CHECK(!idx_type.is<type::Str>(),
                                "String index in aten::__getitem__ means dict input, this is not supported.");
    if (ov::as_type_ptr<ov::op::util::FrameworkNode>(input.get_node_shared_ptr())) {
        PYTORCH_OP_CONVERSION_CHECK(
            !cast_fw_node(input.get_node_shared_ptr(), {"aten::split", "aten::chunk", "aten::unsafe_chunk"}),
            "special case for aten::__getitem__");
        const auto&& list_elems = get_list_as_outputs(input);
        auto getitem_idx = context.const_input<int64_t>(1);
        if (getitem_idx < 0) {
            getitem_idx += list_elems.size();
        }
        PYTORCH_OP_CONVERSION_CHECK(getitem_idx < static_cast<int64_t>(list_elems.size()),
                                    "Index: ",
                                    getitem_idx,
                                    " is out of bounds of input list of len: ",
                                    list_elems.size());
        return {list_elems.at(getitem_idx)};
    }
    if (ov::as_type_ptr<v0::Parameter>(input.get_node_shared_ptr())) {
        const auto& outside_input_node = context.get_input_from_visible_context(0).get_node_shared_ptr();
        PYTORCH_OP_CONVERSION_CHECK(!ov::as_type_ptr<v5::Loop>(outside_input_node),
                                    "Unsupported case: aten::__getitem__ is inside the body, and input is Loop.");
    }
    auto getitem_idx = context.get_input(1);
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<v8::Gather>(input, getitem_idx, zero))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov