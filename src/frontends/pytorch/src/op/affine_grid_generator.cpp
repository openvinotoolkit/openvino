// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// aten::affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> Tensor
// theta: (N, 2, 3) for 2D, (N, 3, 4) for 3D
// size: (N, C, H, W) for 2D, (N, C, D, H, W) for 3D
// output: (N, H, W, 2) for 2D, (N, D, H, W, 3) for 3D
OutputVector translate_affine_grid_generator(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    
    auto theta = context.get_input(0);  // (N, 2, 3) for 2D
    auto size_input = context.const_input<std::vector<int64_t>>(1);
    auto align_corners = context.const_input<bool>(2);
    
    // Determine if 2D or 3D based on size length
    bool is_2d = (size_input.size() == 4);  // N, C, H, W
    PYTORCH_OP_CONVERSION_CHECK(is_2d, "Only 2D affine_grid_generator is currently supported");
    
    auto element_type = theta.get_element_type();
    auto const_0 = v0::Constant::create(element::i64, Shape{}, {0});
    auto const_1 = v0::Constant::create(element::i64, Shape{}, {1});
    auto const_neg1 = v0::Constant::create(element::i64, Shape{1}, {-1});
    
    // Extract dimensions from size
    int64_t N = size_input[0];
    int64_t H = size_input[2];
    int64_t W = size_input[3];
    
    // Create coordinate grids
    // For align_corners=True:  linspace(-1, 1, steps)
    // For align_corners=False: linspace(-1 + 1/steps, 1 - 1/steps, steps)
    
    auto create_linspace = [&](int64_t steps) -> Output<Node> {
        auto steps_f = static_cast<double>(steps);
        double start, end;
        
        if (align_corners) {
            start = -1.0;
            end = 1.0;
        } else {
            start = -1.0 + 1.0 / steps_f;
            end = 1.0 - 1.0 / steps_f;
        }
        
        if (steps == 1) {
            return context.mark_node(v0::Constant::create(element_type, Shape{1}, {0.0}));
        }
        
        double step = (end - start) / (steps - 1);
        auto start_const = v0::Constant::create(element_type, Shape{}, {start});
        auto step_const = v0::Constant::create(element_type, Shape{}, {step});
        auto limit_const = v0::Constant::create(element_type, Shape{}, {end + step / 2});  // Ensure we get all steps
        
        return context.mark_node(std::make_shared<v4::Range>(start_const, limit_const, step_const, element_type));
    };
    
    // Create y coordinates (H values)
    auto y_coords = create_linspace(H);  // Shape: (H,)
    // Create x coordinates (W values)  
    auto x_coords = create_linspace(W);  // Shape: (W,)
    
    // Create meshgrid: expand to (H, W) each
    // y_grid: (H, 1) -> broadcast to (H, W)
    // x_grid: (1, W) -> broadcast to (H, W)
    auto h_shape = v0::Constant::create(element::i64, Shape{2}, {H, 1});
    auto w_shape = v0::Constant::create(element::i64, Shape{2}, {1, W});
    auto hw_shape = v0::Constant::create(element::i64, Shape{2}, {H, W});
    
    auto y_reshaped = context.mark_node(std::make_shared<v1::Reshape>(y_coords, h_shape, false));
    auto x_reshaped = context.mark_node(std::make_shared<v1::Reshape>(x_coords, w_shape, false));
    
    auto y_grid = context.mark_node(std::make_shared<v3::Broadcast>(y_reshaped, hw_shape));  // (H, W)
    auto x_grid = context.mark_node(std::make_shared<v3::Broadcast>(x_reshaped, hw_shape));  // (H, W)
    
    // Flatten grids for matrix multiplication
    auto y_flat = context.mark_node(std::make_shared<v1::Reshape>(y_grid, const_neg1, false));  // (H*W,)
    auto x_flat = context.mark_node(std::make_shared<v1::Reshape>(x_grid, const_neg1, false));  // (H*W,)
    
    // Create ones for homogeneous coordinates
    auto ones = context.mark_node(v0::Constant::create(element_type, Shape{H * W}, std::vector<double>(H * W, 1.0)));
    
    // Stack to create (H*W, 3) grid with [x, y, 1] - note: x first, then y for PyTorch convention
    auto coord_stack_shape = v0::Constant::create(element::i64, Shape{2}, {static_cast<int64_t>(H * W), 1});
    auto x_col = context.mark_node(std::make_shared<v1::Reshape>(x_flat, coord_stack_shape, false));
    auto y_col = context.mark_node(std::make_shared<v1::Reshape>(y_flat, coord_stack_shape, false));
    auto ones_col = context.mark_node(std::make_shared<v1::Reshape>(ones, coord_stack_shape, false));
    
    // Concatenate to (H*W, 3): [x, y, 1] for each position
    auto grid_coords = context.mark_node(std::make_shared<v0::Concat>(OutputVector{x_col, y_col, ones_col}, 1));  // (H*W, 3)
    
    // Add batch dimension: (1, H*W, 3)
    auto grid_batched = context.mark_node(std::make_shared<v0::Unsqueeze>(grid_coords, const_0));
    
    // Transpose theta from (N, 2, 3) to (N, 3, 2) for matmul
    auto transpose_order = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
    auto theta_transposed = context.mark_node(std::make_shared<v1::Transpose>(theta, transpose_order));  // (N, 3, 2)
    
    // Matrix multiplication: (1, H*W, 3) @ (N, 3, 2) -> (N, H*W, 2)
    auto result = context.mark_node(std::make_shared<v0::MatMul>(grid_batched, theta_transposed, false, false));
    
    // Reshape to (N, H, W, 2)
    auto output_shape = v0::Constant::create(element::i64, Shape{4}, {N, H, W, 2});
    auto output = context.mark_node(std::make_shared<v1::Reshape>(result, output_shape, false));
    
    return {output};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
