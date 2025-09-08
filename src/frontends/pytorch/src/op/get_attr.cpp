// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_get_attr(const NodeContext& context) {
    auto decoder = context.get_decoder();
    auto res = decoder->try_decode_get_attr();
    PYTORCH_OP_CONVERSION_CHECK(res.size() > 0,
                                "Failed to obtain data from GetAttr with output tensor name: ",
                                decoder->get_output_debug_name(0));
    if (res.size() == 1) {
        auto node = res[0].get_node();
        if (node && node->get_friendly_name() != node->get_name()) {
            res[0].add_names({node->get_friendly_name()});
        }
        auto dtype = decoder->get_output_type(0);
        if (dtype.is<type::Complex>()) {
            // Add complex mark to complex constant
            res = {context.mark_node(std::make_shared<ComplexTypeMark>(res[0], res[0].get_element_type()))};
        }
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
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
