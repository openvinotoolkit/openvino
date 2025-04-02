// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/pytorch/node_context.hpp>
#include <openvino/op/constant.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
OutputVector st(const NodeContext& context) {
    
    auto input_tensor = context.get_input(0);
    
    auto const_node = ov::as_type_ptr<op::v0::Constant>(input_tensor.get_node_shared_ptr());
    
    // Check if the input is a constant.
    if (!const_node) {
        throw std::runtime_error("prim::tolist is only supported for constant tensors.");
    }
    
    // Extract the constant data.
    std::vector<int32_t> values = const_node->cast_vector<int32_t>();
    
    auto output_node = context.mark_node(
        op::v0::Constant::create(element::i32, Shape{values.size()}, values)
    );
    
    return {output_node};
    
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov