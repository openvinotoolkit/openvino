// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/attribute.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

namespace detail {

std::shared_ptr<ov::Node> to_scalar(const ov::Output<ov::Node>& input) {
    const auto zero = v0::Constant::create(ov::element::i32, Shape{1}, {0});
    const auto one = v0::Constant::create(ov::element::i32, Shape{1}, {1});
    auto sliced = std::make_shared<v8::Slice>(input, zero, one, one, zero);
    auto squeeze_axes = v0::Constant::create(ov::element::i32, Shape{1}, {0});
    return std::make_shared<v0::Squeeze>(sliced, squeeze_axes);
}

std::shared_ptr<ov::Node> construct_original_grid(const ov::Output<ov::Node>& data_size,
                                                  bool align_corners) {
    const auto const_zero = v0::Constant::create(ov::element::f32, Shape{}, {0.0f});
    const auto const_one = v0::Constant::create(ov::element::f32, Shape{}, {1.0f});
    const auto const_two = v0::Constant::create(ov::element::f32, Shape{}, {2.0f});
    const auto const_minus_one = v0::Constant::create(ov::element::f32, Shape{}, {-1.0f});
    const auto const_half = v0::Constant::create(ov::element::f32, Shape{}, {0.5f});

    const auto const_zero_i32 = v0::Constant::create(ov::element::i32, Shape{1}, {0});
    const auto const_one_i32 = v0::Constant::create(ov::element::i32, Shape{1}, {1});
    const auto const_two_i32 = v0::Constant::create(ov::element::i32, Shape{1}, {2});

    const auto dim_0 = std::make_shared<v8::Slice>(data_size, const_zero_i32, const_one_i32, const_one_i32, const_zero_i32);
    const auto dim_1 = std::make_shared<v8::Slice>(data_size, const_one_i32, const_two_i32, const_one_i32, const_zero_i32);

    const auto dim_0_f = std::make_shared<v0::Convert>(dim_0, ov::element::f32);
    const auto dim_1_f = std::make_shared<v0::Convert>(dim_1, ov::element::f32);

    const auto dim_0_scalar = to_scalar(dim_0_f);
    const auto dim_1_scalar = to_scalar(dim_1_f);

    std::shared_ptr<ov::Node> range_0, range_1;

    if (align_corners) {
        const auto dim_0_minus_one = std::make_shared<v1::Subtract>(dim_0_scalar, const_one);
        const auto dim_1_minus_one = std::make_shared<v1::Subtract>(dim_1_scalar, const_one);

        const auto step_0 = std::make_shared<v1::Divide>(const_two, dim_0_minus_one);
        const auto step_1 = std::make_shared<v1::Divide>(const_two, dim_1_minus_one);

        range_0 = std::make_shared<v4::Range>(const_minus_one, const_one, step_0, ov::element::f32);
        range_1 = std::make_shared<v4::Range>(const_minus_one, const_one, step_1, ov::element::f32);
    } else {
        const auto step_0 = std::make_shared<v1::Divide>(const_two, dim_0_scalar);
        const auto step_1 = std::make_shared<v1::Divide>(const_two, dim_1_scalar);

        const auto step_0_half = std::make_shared<v1::Multiply>(step_0, const_half);
        const auto step_1_half = std::make_shared<v1::Multiply>(step_1, const_half);

        const auto start_0 = std::make_shared<v1::Add>(const_minus_one, step_0_half);
        const auto start_1 = std::make_shared<v1::Add>(const_minus_one, step_1_half);

        range_0 = std::make_shared<v4::Range>(start_0, const_one, step_0, ov::element::f32);
        range_1 = std::make_shared<v4::Range>(start_1, const_one, step_1, ov::element::f32);
    }

    const auto dim_0_i32 = std::make_shared<v0::Convert>(dim_0, ov::element::i32);
    const auto dim_1_i32 = std::make_shared<v0::Convert>(dim_1, ov::element::i32);

    const auto shape_2d_y = std::make_shared<v0::Concat>(ov::OutputVector{dim_0_i32, const_one_i32}, 0);
    const auto shape_2d_x = std::make_shared<v0::Concat>(ov::OutputVector{const_one_i32, dim_1_i32}, 0);
    const auto shape_2d_broadcast = std::make_shared<v0::Concat>(ov::OutputVector{dim_0_i32, dim_1_i32}, 0);

    const auto y_reshaped = std::make_shared<v1::Reshape>(range_0, shape_2d_y, false);
    const auto x_reshaped = std::make_shared<v1::Reshape>(range_1, shape_2d_x, false);

    const auto y_broadcast = std::make_shared<v3::Broadcast>(y_reshaped, shape_2d_broadcast);
    const auto x_broadcast = std::make_shared<v3::Broadcast>(x_reshaped, shape_2d_broadcast);

    const auto ones = v0::Constant::create(ov::element::f32, Shape{1}, {1.0f});
    const auto ones_broadcast = std::make_shared<v3::Broadcast>(ones, shape_2d_broadcast);

    const auto axis_2 = v0::Constant::create(ov::element::i32, Shape{1}, {2});
    const auto grid_homo = std::make_shared<v0::Concat>(ov::OutputVector{
        std::make_shared<v0::Unsqueeze>(x_broadcast, axis_2),
        std::make_shared<v0::Unsqueeze>(y_broadcast, axis_2),
        std::make_shared<v0::Unsqueeze>(ones_broadcast, axis_2)
    }, 2);

    return grid_homo;
}

std::shared_ptr<ov::Node> apply_affine_transform(const ov::Output<ov::Node>& theta,
                                                 const ov::Output<ov::Node>& grid_homo) {
    auto tshape = std::make_shared<v3::ShapeOf>(theta);
    auto gshape = std::make_shared<v3::ShapeOf>(grid_homo);
    auto i0 = v0::Constant::create(ov::element::i32, Shape{1}, {0});
    auto i1 = v0::Constant::create(ov::element::i32, Shape{1}, {1});
    auto i2 = v0::Constant::create(ov::element::i32, Shape{1}, {2});

    auto N = std::make_shared<v8::Slice>(tshape, i0, i1, i1, i0);
    auto H = std::make_shared<v8::Slice>(gshape, i0, i1, i1, i0);
    auto W = std::make_shared<v8::Slice>(gshape, i1, i2, i1, i0);

    auto HW = std::make_shared<v1::Multiply>(H, W);
    auto HW_i64 = std::make_shared<v0::Convert>(HW, ov::element::i64);
    auto three = v0::Constant::create(ov::element::i64, Shape{1}, {3});
    auto resh = std::make_shared<v0::Concat>(OutputVector{HW_i64, three}, 0);
    auto flat = std::make_shared<v1::Reshape>(grid_homo, resh, false);
    auto perm = v0::Constant::create(ov::element::i32, Shape{2}, {1, 0});
    auto trans = std::make_shared<v1::Transpose>(flat, perm);

    auto mm = std::make_shared<v0::MatMul>(theta, trans);
    auto perm2 = v0::Constant::create(ov::element::i32, Shape{3}, {0, 2, 1});
    auto trans2 = std::make_shared<v1::Transpose>(mm, perm2);

    auto two = v0::Constant::create(ov::element::i64, Shape{1}, {2});
    auto N_i64 = std::make_shared<v0::Convert>(N, ov::element::i64);
    auto H_i64 = std::make_shared<v0::Convert>(H, ov::element::i64);
    auto W_i64 = std::make_shared<v0::Convert>(W, ov::element::i64);
    auto final_shape = std::make_shared<v0::Concat>(OutputVector{N_i64, H_i64, W_i64, two}, 0);

    return std::make_shared<v1::Reshape>(trans2, final_shape, false);
}

ov::OutputVector affine_grid(const ov::OutputVector& inputs, const bool align_corners) {
    const auto& theta = inputs[0];
    const auto& size = inputs[1];

    const auto const_two_i32 = v0::Constant::create(ov::element::i32, Shape{1}, {2});
    const auto const_one_i32 = v0::Constant::create(ov::element::i32, Shape{1}, {1});
    const auto const_zero_i32 = v0::Constant::create(ov::element::i32, Shape{1}, {0});

    const auto size_shape = std::make_shared<v3::ShapeOf>(size);
    const auto size_len_scalar = std::make_shared<v8::Slice>(size_shape, const_zero_i32, const_one_i32, const_one_i32, const_zero_i32);
    
    const auto size_len_i64 = std::make_shared<v0::Convert>(size_len_scalar, ov::element::i64);
    const auto size_len_reshaped = std::make_shared<v1::Reshape>(size_len_i64, 
        v0::Constant::create(ov::element::i64, Shape{1}, {1}), false);
    
    const auto data_size = std::make_shared<v8::Slice>(size, const_two_i32, size_len_reshaped, const_one_i32, const_zero_i32);

    const auto original_grid = construct_original_grid(data_size, align_corners);
    return {apply_affine_transform(theta, original_grid)};
}

}  // namespace detail

ov::OutputVector affine_grid(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);
    const auto inputs = node.get_ov_inputs();
    const auto align_corners = node.get_attribute_value<int64_t>("align_corners", 0) != 0;
    return detail::affine_grid(inputs, align_corners);
}

ONNX_OP("AffineGrid", OPSET_SINCE(1), ai_onnx::opset_1::affine_grid);

}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov