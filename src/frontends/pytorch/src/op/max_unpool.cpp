// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor
// aten::max_unpool2d.out(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
OutputVector translate_max_unpool2d(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    
    auto input = context.get_input(0);       // Pooled values
    auto indices = context.get_input(1);     // Indices from max_pool2d
    // output_size is context.get_input(2)   // Target [H, W]
    
    auto const_0 = v0::Constant::create(element::i64, Shape{}, {0});
    auto const_1 = v0::Constant::create(element::i64, Shape{}, {1});
    auto const_neg1 = v0::Constant::create(element::i64, Shape{1}, {-1});
    
    // Get output_size as [H, W]
    auto output_size = context.const_input<std::vector<int64_t>>(2);
    int64_t out_h = output_size[0];
    int64_t out_w = output_size[1];
    
    // Get input shape [N, C, h, w] or [C, h, w]
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i64));
    auto input_rank = input.get_partial_shape().rank();
    
    bool is_batched = input_rank.is_static() && input_rank.get_length() == 4;
    
    // Flatten input and indices for ScatterNDUpdate
    auto flat_input = context.mark_node(std::make_shared<v1::Reshape>(input, const_neg1, false));
    auto flat_indices = context.mark_node(std::make_shared<v1::Reshape>(indices, const_neg1, false));
    
    // Add extra dimension for ScatterNDUpdate: [N] -> [N, 1]
    auto scatter_indices = context.mark_node(std::make_shared<v0::Unsqueeze>(flat_indices, const_1));
    
    // Compute total output size
    int64_t batch_size = 1;
    int64_t channels = 1;
    if (is_batched) {
        // For batched input, we need to handle N and C dimensions
        // Output shape: [N, C, out_h, out_w]
        // We create zeros of shape [N * C * out_h * out_w] then reshape
    }
    
    // Create output shape tensor
    Output<Node> output_shape_node;
    if (is_batched) {
        // Get N and C from input shape
        auto n_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape, 
            v0::Constant::create(element::i64, Shape{1}, {0}), const_0));
        auto c_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape,
            v0::Constant::create(element::i64, Shape{1}, {1}), const_0));
        auto hw_dims = v0::Constant::create(element::i64, Shape{2}, {out_h, out_w});
        output_shape_node = context.mark_node(std::make_shared<v0::Concat>(
            OutputVector{n_dim, c_dim, hw_dims}, 0));
    } else {
        // Unbatched: [C, out_h, out_w]
        auto c_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape,
            v0::Constant::create(element::i64, Shape{1}, {0}), const_0));
        auto hw_dims = v0::Constant::create(element::i64, Shape{2}, {out_h, out_w});
        output_shape_node = context.mark_node(std::make_shared<v0::Concat>(
            OutputVector{c_dim, hw_dims}, 0));
    }
    
    // Compute flat output size for zeros tensor
    auto flat_output_size = context.mark_node(std::make_shared<v1::ReduceProd>(output_shape_node, const_0, true));
    
    // Create zeros tensor
    auto zeros = context.mark_node(std::make_shared<v3::Broadcast>(
        v0::Constant::create(input.get_element_type(), Shape{}, {0}),
        flat_output_size));
    
    // Scatter input values into zeros at index positions
    auto scattered = context.mark_node(std::make_shared<v3::ScatterNDUpdate>(
        zeros, scatter_indices, flat_input));
    
    // Reshape to output shape
    auto result = context.mark_node(std::make_shared<v1::Reshape>(scattered, output_shape_node, false));
    
    return {result};
}

OutputVector translate_max_unpool3d(const NodeContext& context) {
    // TODO: Implement 3D version
    FRONT_END_THROW("max_unpool3d is not yet implemented");
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
