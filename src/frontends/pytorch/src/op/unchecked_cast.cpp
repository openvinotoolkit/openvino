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

OutputVector translate_unchecked_cast(const NodeContext& context);

OutputVector translate_unchecked_cast(const NodeContext& context) {
    //Ensure the input exists and is a tensor
    FRONT_END_OP_CONVERSION_CHECK(context.get_input_size() >= 1,
                                   "translate_unchecked_cast expects at least 1 input, but got: ",
                                   context.get_input_size());
    
    //Get the input tensor and the target type                               
    auto input = context.get_input(0);
    auto target_type = context.get_attribute<ov::element::Type>("dtype");

    //Use existing OpenVino Convert op to convert the input tensor to the target type
    auto convert_node = std::make_shared<ov::op::v0::Convert>(input, target_type);

    //Mark this node for graph tracking
    context.mark_node(convert_node);

    //Return the converted node as output
    return {convert_node};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov