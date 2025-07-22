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

std::shared_ptr<ov::Node> construct_original_grid_2d(const ov::Output<ov::Node>& data_size,
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
        const auto is_dim_0_one = std::make_shared<v1::Equal>(dim_0_scalar, const_one);
        const auto is_dim_1_one = std::make_shared<v1::Equal>(dim_1_scalar, const_one);

        const auto zero_range = v0::Constant::create(ov::element::f32, Shape{1}, {0.0f});

        const auto dim_0_minus_one = std::make_shared<v1::Subtract>(dim_0_scalar, const_one);
        const auto dim_1_minus_one = std::make_shared<v1::Subtract>(dim_1_scalar, const_one);

        const auto safe_denom_0 = std::make_shared<v1::Select>(is_dim_0_one, const_one, dim_0_minus_one);
        const auto safe_denom_1 = std::make_shared<v1::Select>(is_dim_1_one, const_one, dim_1_minus_one);
        
        const auto step_0 = std::make_shared<v1::Divide>(const_two, safe_denom_0);
        const auto step_1 = std::make_shared<v1::Divide>(const_two, safe_denom_1);

        const auto epsilon = v0::Constant::create(ov::element::f32, Shape{}, {1e-6f});
        const auto one_plus_eps = std::make_shared<v1::Add>(const_one, epsilon);

        auto range_0_normal = std::make_shared<v4::Range>(const_minus_one, one_plus_eps, step_0, ov::element::f32);
        auto range_1_normal = std::make_shared<v4::Range>(const_minus_one, one_plus_eps, step_1, ov::element::f32);

        range_0 = std::make_shared<v1::Select>(is_dim_0_one, zero_range, range_0_normal);
        range_1 = std::make_shared<v1::Select>(is_dim_1_one, zero_range, range_1_normal);
        
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

    auto range_0_shape = std::make_shared<v3::ShapeOf>(range_0);
    auto range_1_shape = std::make_shared<v3::ShapeOf>(range_1);

    auto one = v0::Constant::create(ov::element::i64, Shape{1}, {1});

    auto shape_2d_y = std::make_shared<v0::Concat>(OutputVector{range_0_shape, one}, 0); // [?,1]
    auto shape_2d_x = std::make_shared<v0::Concat>(OutputVector{one, range_1_shape}, 0); // [1,?]
    auto shape_2d_broadcast = std::make_shared<v0::Concat>(OutputVector{range_0_shape, range_1_shape}, 0); // [?,?]

    const auto y_reshaped = std::make_shared<v1::Reshape>(range_0, shape_2d_y, false);
    const auto x_reshaped = std::make_shared<v1::Reshape>(range_1, shape_2d_x, false);

    const auto y_broadcast = std::make_shared<v3::Broadcast>(y_reshaped, shape_2d_broadcast);
    const auto x_broadcast = std::make_shared<v3::Broadcast>(x_reshaped, shape_2d_broadcast);

    const auto ones = v0::Constant::create(ov::element::f32, Shape{1}, {1.0f});
    const auto ones_broadcast = std::make_shared<v3::Broadcast>(ones, shape_2d_broadcast);

    const auto axis_2 = v0::Constant::create(ov::element::i32, Shape{1}, {2});
    const auto grid_homo = std::make_shared<v0::Concat>(OutputVector{
        std::make_shared<v0::Unsqueeze>(x_broadcast, axis_2),
        std::make_shared<v0::Unsqueeze>(y_broadcast, axis_2),
        std::make_shared<v0::Unsqueeze>(ones_broadcast, axis_2)
    }, 2);

    return grid_homo;
}

std::shared_ptr<ov::Node> construct_original_grid_3d(const ov::Output<ov::Node>& data_size,
                                                     bool align_corners) {
    const auto f32 = ov::element::f32;
    const auto i32 = ov::element::i32;
    const auto i64 = ov::element::i64;

    const auto const_zero = v0::Constant::create(f32, Shape{}, {0.0f});
    const auto const_one = v0::Constant::create(f32, Shape{}, {1.0f});
    const auto const_two = v0::Constant::create(f32, Shape{}, {2.0f});
    const auto const_minus_one = v0::Constant::create(f32, Shape{}, {-1.0f});
    const auto const_half = v0::Constant::create(f32, Shape{}, {0.5f});

    const auto i0 = v0::Constant::create(i32, Shape{1}, {0});
    const auto i1 = v0::Constant::create(i32, Shape{1}, {1});
    const auto i2 = v0::Constant::create(i32, Shape{1}, {2});
    const auto i3 = v0::Constant::create(i32, Shape{1}, {3});
    const auto i64_one = v0::Constant::create(i64, Shape{1}, {1});

    const auto d = std::make_shared<v8::Slice>(data_size, i0, i1, i1, i0);
    const auto h = std::make_shared<v8::Slice>(data_size, i1, i2, i1, i0);
    const auto w = std::make_shared<v8::Slice>(data_size, i2, i3, i1, i0);

    const auto d_f = std::make_shared<v0::Convert>(d, f32);
    const auto h_f = std::make_shared<v0::Convert>(h, f32);
    const auto w_f = std::make_shared<v0::Convert>(w, f32);

    const auto d_scalar = to_scalar(d_f);
    const auto h_scalar = to_scalar(h_f);
    const auto w_scalar = to_scalar(w_f);

    std::shared_ptr<ov::Node> range_d, range_h, range_w;

    if (align_corners) {
        const auto is_d_one = std::make_shared<v1::Equal>(d_scalar, const_one);
        const auto is_h_one = std::make_shared<v1::Equal>(h_scalar, const_one);
        const auto is_w_one = std::make_shared<v1::Equal>(w_scalar, const_one);

        const auto zero_range = v0::Constant::create(f32, Shape{1}, {0.0f});
        
        const auto d_minus1 = std::make_shared<v1::Subtract>(d_f, const_one);
        const auto h_minus1 = std::make_shared<v1::Subtract>(h_f, const_one);
        const auto w_minus1 = std::make_shared<v1::Subtract>(w_f, const_one);

        const auto safe_denom_d = std::make_shared<v1::Select>(is_d_one, const_one, to_scalar(d_minus1));
        const auto safe_denom_h = std::make_shared<v1::Select>(is_h_one, const_one, to_scalar(h_minus1));
        const auto safe_denom_w = std::make_shared<v1::Select>(is_w_one, const_one, to_scalar(w_minus1));

        const auto step_d = std::make_shared<v1::Divide>(const_two, safe_denom_d);
        const auto step_h = std::make_shared<v1::Divide>(const_two, safe_denom_h);
        const auto step_w = std::make_shared<v1::Divide>(const_two, safe_denom_w);

        const auto epsilon = v0::Constant::create(f32, Shape{}, {1e-6f});
        const auto one_plus_eps = std::make_shared<v1::Add>(const_one, epsilon);

        auto range_d_normal = std::make_shared<v4::Range>(const_minus_one, one_plus_eps, step_d, f32);
        auto range_h_normal = std::make_shared<v4::Range>(const_minus_one, one_plus_eps, step_h, f32);
        auto range_w_normal = std::make_shared<v4::Range>(const_minus_one, one_plus_eps, step_w, f32);
        
        range_d = std::make_shared<v1::Select>(is_d_one, zero_range, range_d_normal);
        range_h = std::make_shared<v1::Select>(is_h_one, zero_range, range_h_normal);
        range_w = std::make_shared<v1::Select>(is_w_one, zero_range, range_w_normal);
        
    } else {
        const auto step_d = std::make_shared<v1::Divide>(const_two, d_scalar);
        const auto step_h = std::make_shared<v1::Divide>(const_two, h_scalar);
        const auto step_w = std::make_shared<v1::Divide>(const_two, w_scalar);

        const auto half_step_d = std::make_shared<v1::Multiply>(step_d, const_half);
        const auto half_step_h = std::make_shared<v1::Multiply>(step_h, const_half);
        const auto half_step_w = std::make_shared<v1::Multiply>(step_w, const_half);

        const auto start_d = std::make_shared<v1::Add>(const_minus_one, half_step_d);
        const auto start_h = std::make_shared<v1::Add>(const_minus_one, half_step_h);
        const auto start_w = std::make_shared<v1::Add>(const_minus_one, half_step_w);

        range_d = std::make_shared<v4::Range>(start_d, const_one, step_d, f32);
        range_h = std::make_shared<v4::Range>(start_h, const_one, step_h, f32);
        range_w = std::make_shared<v4::Range>(start_w, const_one, step_w, f32);
    }

    const auto shape_d = std::make_shared<v3::ShapeOf>(range_d);
    const auto shape_h = std::make_shared<v3::ShapeOf>(range_h);
    const auto shape_w = std::make_shared<v3::ShapeOf>(range_w);

    const auto shape_d_b = std::make_shared<v0::Concat>(OutputVector{shape_d, i64_one, i64_one}, 0);
    const auto shape_h_b = std::make_shared<v0::Concat>(OutputVector{i64_one, shape_h, i64_one}, 0);
    const auto shape_w_b = std::make_shared<v0::Concat>(OutputVector{i64_one, i64_one, shape_w}, 0);
    const auto shape_dhw = std::make_shared<v0::Concat>(OutputVector{shape_d, shape_h, shape_w}, 0);

    const auto grid_d = std::make_shared<v1::Reshape>(range_d, shape_d_b, false);
    const auto grid_h = std::make_shared<v1::Reshape>(range_h, shape_h_b, false);
    const auto grid_w = std::make_shared<v1::Reshape>(range_w, shape_w_b, false);

    const auto grid_d_b = std::make_shared<v3::Broadcast>(grid_d, shape_dhw);
    const auto grid_h_b = std::make_shared<v3::Broadcast>(grid_h, shape_dhw);
    const auto grid_w_b = std::make_shared<v3::Broadcast>(grid_w, shape_dhw);

    const auto ones = v0::Constant::create(f32, Shape{1}, {1.0f});
    const auto ones_b = std::make_shared<v3::Broadcast>(ones, shape_dhw);

    const auto axis3 = v0::Constant::create(i32, Shape{1}, {3});
    const auto grid_homo = std::make_shared<v0::Concat>(
        OutputVector{
            std::make_shared<v0::Unsqueeze>(grid_w_b, axis3),
            std::make_shared<v0::Unsqueeze>(grid_h_b, axis3),
            std::make_shared<v0::Unsqueeze>(grid_d_b, axis3),
            std::make_shared<v0::Unsqueeze>(ones_b, axis3)},
        3);

    return grid_homo;
}

std::shared_ptr<ov::Node> apply_affine_transform_2d(const ov::Output<ov::Node>& theta,
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

std::shared_ptr<ov::Node> apply_affine_transform_3d(const ov::Output<ov::Node>& theta,
                                                    const ov::Output<ov::Node>& grid_homo) {
    std::cout << "[AffineGrid 3D] Starting affine transform\n";

    auto tshape = std::make_shared<ov::op::v3::ShapeOf>(theta);
    auto gshape = std::make_shared<ov::op::v3::ShapeOf>(grid_homo);

    auto i0 = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
    auto i1 = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
    auto i2 = ov::op::v0::Constant::create(ov::element::i32, {1}, {2});
    auto i3 = ov::op::v0::Constant::create(ov::element::i32, {1}, {3});

    auto N = std::make_shared<ov::op::v8::Slice>(tshape, i0, i1, i1, i0);
    auto D = std::make_shared<ov::op::v8::Slice>(gshape, i0, i1, i1, i0);
    auto H = std::make_shared<ov::op::v8::Slice>(gshape, i1, i2, i1, i0);
    auto W = std::make_shared<ov::op::v8::Slice>(gshape, i2, i3, i1, i0);

    std::cout << "[AffineGrid 3D] Extracted N, D, H, W\n";

    auto DH = std::make_shared<ov::op::v1::Multiply>(D, H);
    auto DHW = std::make_shared<ov::op::v1::Multiply>(DH, W);
    auto DHW_i64 = std::make_shared<ov::op::v0::Convert>(DHW, ov::element::i64);
    auto four = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {4});
    auto reshape_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{DHW_i64, four}, 0);

    auto flat = std::make_shared<ov::op::v1::Reshape>(grid_homo, reshape_shape, false);

    std::cout << "[AffineGrid 3D] Reshaped grid_homo to [DHW, 4]\n";

    auto perm1 = ov::op::v0::Constant::create(ov::element::i32, Shape{2}, {1, 0});
    auto trans = std::make_shared<ov::op::v1::Transpose>(flat, perm1);

    auto mm = std::make_shared<ov::op::v0::MatMul>(theta, trans);

    auto perm2 = ov::op::v0::Constant::create(ov::element::i32, Shape{3}, {0, 2, 1});
    auto trans2 = std::make_shared<ov::op::v1::Transpose>(mm, perm2);

    std::cout << "[AffineGrid 3D] Affine transform complete -> shape [N, DHW, 3]\n";

    auto N_i64 = std::make_shared<ov::op::v0::Convert>(N, ov::element::i64);
    auto D_i64 = std::make_shared<ov::op::v0::Convert>(D, ov::element::i64);
    auto H_i64 = std::make_shared<ov::op::v0::Convert>(H, ov::element::i64);
    auto W_i64 = std::make_shared<ov::op::v0::Convert>(W, ov::element::i64);
    auto three = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {3});

    auto final_shape = std::make_shared<ov::op::v0::Concat>(
        OutputVector{N_i64, D_i64, H_i64, W_i64, three}, 0
    );

    std::cout << "[AffineGrid 3D] Reshaping final output to [N, D, H, W, 3]\n";

    return std::make_shared<ov::op::v1::Reshape>(trans2, final_shape, false);
}

ov::OutputVector affine_grid(const ov::OutputVector& inputs, const bool align_corners) {
    const auto& theta = inputs[0]; 
    const auto& size = inputs[1];  

    const auto& size_pshape = size.get_partial_shape();

    if (size_pshape.rank().is_static()) {
        int64_t rank = size_pshape[0].get_length();

        if (rank == 4) {
            const auto data_size = std::make_shared<v8::Slice>(size,
                v0::Constant::create(ov::element::i32, Shape{1}, {2}),
                v0::Constant::create(ov::element::i32, Shape{1}, {4}),
                v0::Constant::create(ov::element::i32, Shape{1}, {1}),
                v0::Constant::create(ov::element::i32, Shape{1}, {0})
            );
            auto grid = construct_original_grid_2d(data_size, align_corners);
            return {apply_affine_transform_2d(theta, grid)};
        } else if (rank == 5) {
            const auto data_size = std::make_shared<v8::Slice>(size,
                v0::Constant::create(ov::element::i32, Shape{1}, {2}),
                v0::Constant::create(ov::element::i32, Shape{1}, {5}),
                v0::Constant::create(ov::element::i32, Shape{1}, {1}),
                v0::Constant::create(ov::element::i32, Shape{1}, {0})
            );
            auto grid = construct_original_grid_3d(data_size, align_corners);
            return {apply_affine_transform_3d(theta, grid)};
        } else {
            FRONT_END_GENERAL_CHECK(false, "AffineGrid supports only 4D (2D) or 5D (3D) input sizes.");
        }
    } else {
        FRONT_END_GENERAL_CHECK(false, "AffineGrid input 'size' must have a static rank.");
    }
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
