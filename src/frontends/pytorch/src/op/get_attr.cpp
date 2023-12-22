// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_get_attr(const NodeContext& context) {
    auto res = context.get_decoder()->try_decode_get_attr();
    FRONT_END_OP_CONVERSION_CHECK(res.size() > 0,
                                  "Failed to obtain data from GetAttr with output tensor name: ",
                                  context.get_decoder()->get_output_debug_name(0));
    if (res.size() == 1) {
        return res;
    } else {
        // Packed params case
        std::shared_ptr<Node> fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), res, 1);
        add_exception_to_fw_node(
            fw_node,
            "PackedParams represented as FrameworkNode, all contained params represented as inputs to this "
            "node.");
        return {context.mark_node(fw_node)};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
