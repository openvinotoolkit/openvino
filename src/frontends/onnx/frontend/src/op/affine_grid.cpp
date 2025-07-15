// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "exceptions.hpp"
#include "core/operator_set.hpp"
#include "core/attribute.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector affine_grid(const ov::frontend::onnx::Node& node) {
    const auto theta = node.get_ov_inputs().at(0); // [N, 2/3, 3/4]
    const auto size = node.get_ov_inputs().at(1);  // [N, C, H, W] or [N, C, D, H, W]
    const auto align_corners = node.get_attribute_value<int64_t>("align_corners", 0);

    const auto const_zero = v0::Constant::create(element::i64, Shape{1}, {0});
    const auto const_one = v0::Constant::create(element::i64, Shape{1}, {1});
    const auto const_two = v0::Constant::create(element::i64, Shape{1}, {2});
    const auto const_three = v0::Constant::create(element::i64, Shape{1}, {3});

    const auto spatial_rank = theta.get_shape()[1]; // 2 or 3
    const bool is_2d = spatial_rank == 2;

    std::vector<std::shared_ptr<v0::Constant>> indices;
    if (is_2d)
        indices = {v0::Constant::create(element::i64, Shape{}, {2}), v0::Constant::create(element::i64, Shape{}, {3})};
    else
        indices = {v0::Constant::create(element::i64, Shape{}, {2}),
                   v0::Constant::create(element::i64, Shape{}, {3}),
                   v0::Constant::create(element::i64, Shape{}, {4})};

    // Slice H/W or D/H/W
    std::vector<Output<Node>> grid_axes;
    for (size_t i = 0; i < indices.size(); ++i) {
        auto axis = std::make_shared<v8::Slice>(size, indices[i], indices[i] + const_one, const_one);
        axis = std::make_shared<v0::Reshape>(axis, v0::Constant::create(element::i64, Shape{1}, {1}), false);
        auto axis_fp = std::make_shared<v0::Convert>(axis, element::f32);

        auto dim = size.get_shape()[0]; // static input shape for demo, use ShapeOf otherwise
        auto dim_val = std::dynamic_pointer_cast<v0::Constant>(axis)->cast_vector<int64_t>()[0];
        float step, start;
        if (align_corners) {
            step = 2.f / (dim_val - 1);
            start = -1.f;
        } else {
            step = 2.f / dim_val;
            start = -1.f + step / 2.f;
        }

        auto range = std::make_shared<v4::Range>(
            v0::Constant::create(element::f32, Shape{}, {start}),
            v0::Constant::create(element::f32, Shape{}, {1.f + 1e-4f}),
            v0::Constant::create(element::f32, Shape{}, {step}));

        // Shape to match broadcasting later
        Output<Node> reshaped;
        if (is_2d) {
            if (i == 0)
                reshaped = std::make_shared<v1::Reshape>(range, v0::Constant::create(element::i64, Shape{2}, {dim_val, 1}), false);
            else
                reshaped = std::make_shared<v1::Reshape>(range, v0::Constant::create(element::i64, Shape{2}, {1, dim_val}), false);
        } else {
            if (i == 0)
                reshaped = std::make_shared<v1::Reshape>(range, v0::Constant::create(element::i64, Shape{3}, {dim_val, 1, 1}), false);
            else if (i == 1)
                reshaped = std::make_shared<v1::Reshape>(range, v0::Constant::create(element::i64, Shape{3}, {1, dim_val, 1}), false);
            else
                reshaped = std::make_shared<v1::Reshape>(range, v0::Constant::create(element::i64, Shape{3}, {1, 1, dim_val}), false);
        }

        grid_axes.push_back(reshaped);
    }

    // Concat + Homogeneous coordinate
    grid_axes.push_back(v0::Constant::create(element::f32, Shape{}, {1.f}));
    auto original_grid = std::make_shared<v0::Concat>(grid_axes, is_2d ? 2 : 3);

    // Reshape to [X, dim+1], then broadcast for N
    auto flat_shape = is_2d ? Shape{1, -1, 3} : Shape{1, -1, 4};
    auto reshaped_grid = std::make_shared<v1::Reshape>(original_grid, v0::Constant::create(element::i64, Shape{-1}, {-1, is_2d ? 3 : 4}), true);

    auto transposed = std::make_shared<v1::Transpose>(reshaped_grid, v0::Constant::create(element::i64, Shape{2}, {1, 2}));
    auto grid_out = std::make_shared<v0::MatMul>(theta, transposed);

    // Reshape to [N, H, W, 2] or [N, D, H, W, 3]
    Shape final_shape;
    if (is_2d)
        final_shape = Shape{theta.get_shape()[0], grid_axes[0].get_shape()[0], grid_axes[1].get_shape()[1], 2};
    else
        final_shape = Shape{theta.get_shape()[0], grid_axes[0].get_shape()[0], grid_axes[1].get_shape()[1], grid_axes[2].get_shape()[2], 3};

    auto final_reshaped = std::make_shared<v1::Reshape>(grid_out, v0::Constant::create(element::i64, Shape{final_shape.size()}, final_shape), false);
    return {final_reshaped};
}

ONNX_OP("AffineGrid", OPSET_SINCE(1), ai_onnx::opset_1::affine_grid);

}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
