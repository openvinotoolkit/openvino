// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_native_multi_head_attention(const NodeContext& context) {
    /*
    aten::_native_multi_head_attention(
        Tensor query,
        Tensor key,
        Tensor value,
        int64 embed_dim,
        int64 num_head,
        Tensor qkv_weight,
        Tensor qkv_bias,
        Tensor proj_weight,
        Tensor proj_bias,
        Optional[Tensor] mask = None,
        bool need_weights = true,
        bool average_attn_weights = true,
        Optional[int64] mask_type = None
    )
    */
    num_inputs_check(context, 13, 13);
    const auto need_weights = context.const_input<bool>(10);
    const auto average_weights = context.const_input<bool>(11);

    Output<Node> mask;
    int64_t mask_type = 0;
    if (!context.input_is_none(9) && !context.input_is_none(12)) {
        mask = context.get_input(9);
        mask_type = context.const_input<int64_t>(12);
    }

    const auto mha = build_multi_head_attention(context,
                                                context.get_input(0),
                                                context.get_input(1),
                                                context.get_input(2),
                                                context.get_input(3),
                                                context.get_input(4),
                                                context.get_input(5),
                                                context.get_input(6),
                                                context.get_input(7),
                                                context.get_input(8),
                                                mask,
                                                mask_type,
                                                need_weights,
                                                average_weights);

    if (need_weights) {
        return {mha.first, mha.second};
    }
    // When need_weights == false, returns None as a second output
    const auto none = std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs());
    auto attrs = none->get_attrs();
    attrs["none_value"] = "";
    none->set_attrs(attrs);
    return {mha.first, context.mark_node(none)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
