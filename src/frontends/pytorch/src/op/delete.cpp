// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_delete(const NodeContext& context) {
    // aten::Delete.t(t[](a!) self, int idx) -> ()
    // Removes element `idx` from a list (prim::ListConstruct -> SequenceMark). The
    // list is rebuilt without that element and propagated back via mutate_input, so
    // subsequent aten::len / aten::__getitem__ observe the removal. A no-op would be
    // wrong. Only a constant index is supported (the form TorchScript emits for
    // `del lst[i]`); a non-constant index is rejected (aten::Delete has no output,
    // so a framework-node fallback cannot be represented).
    num_inputs_check(context, 2, 2);
    auto idx_const = ov::as_type_ptr<ov::op::v0::Constant>(context.get_input(1).get_node_shared_ptr());
    if (idx_const) {
        auto elems = get_list_as_outputs(context.get_input(0));
        int64_t idx = idx_const->cast_vector<int64_t>()[0];
        int64_t n = static_cast<int64_t>(elems.size());
        if (idx < 0) {
            idx += n;
        }
        if (idx >= 0 && idx < n) {
            elems.erase(elems.begin() + idx);
        }
        ov::OutputVector remaining(elems.begin(), elems.end());
        auto new_list = context.mark_node(make_list_construct(remaining));
        context.mutate_input(0, new_list);
        return {};
    }
    PYTORCH_OP_CONVERSION_CHECK(false, "aten::Delete is only supported with a constant index.");
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
