// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_quantized_conv2d_prepack(const NodeContext& context) {
    // this operation packs conv parameters into a special node
    // the signature is: quantized::conv2d_prepack(weight, bias, stride, padding, dilation, groups)
    num_inputs_check(context, 6, 6);
    
    auto weight = context.get_input(0);
    auto bias = context.get_input(1);
    auto stride = context.get_input(2);
    auto padding = context.get_input(3);
    auto dilation = context.get_input(4);
    auto groups = context.get_input(5);
    
    // create framework node that stores all parameters as inputs
    // this matches what quantized::conv2d expects from prim::GetAttr
    auto packed_node = std::make_shared<PtFrameworkNode>(
        context.get_decoder(),
        OutputVector{weight, bias, stride, padding, dilation, groups},
        1  // single output
    );
    
    // set the op type to prim::GetAttr so quantized::conv2d recognizes it
    auto& attrs = packed_node->get_attrs();
    attrs[PtFrameworkNode::op_type_key] = "prim::GetAttr";
    
    return {context.mark_node(packed_node)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
