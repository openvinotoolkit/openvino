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
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

namespace detail {

struct GridConstants {
    std::shared_ptr<ov::Node> zero, one, two, minus_one, half, epsilon, zero_range, one_plus_eps;

    GridConstants() {
        zero = v0::Constant::create(ov::element::f32, Shape{}, {0.0f});
        one = v0::Constant::create(ov::element::f32, Shape{}, {1.0f});
        two = v0::Constant::create(ov::element::f32, Shape{}, {2.0f});
        minus_one = v0::Constant::create(ov::element::f32, Shape{}, {-1.0f});
        half = v0::Constant::create(ov::element::f32, Shape{}, {0.5f});
        epsilon = v0::Constant::create(ov::element::f32, Shape{}, {1e-6f});
        zero_range = v0::Constant::create(ov::element::f32, Shape{1}, {0.0f});
        one_plus_eps = std::make_shared<v1::Add>(one, epsilon);
    }
};

std::shared_ptr<ov::Node> to_scalar(const ov::Output<ov::Node>& input) {
    const auto zero = v0::Constant::create(ov::element::i32, Shape{1}, {0});
    const auto one = v0::Constant::create(ov::element::i32, Shape{1}, {1});
    auto sliced = std::make_shared<v8::Slice>(input, zero, one, one, zero);
    auto squeeze_axes = v0::Constant::create(ov::element::i32, Shape{1}, {0});
    return std::make_shared<v0::Squeeze>(sliced, squeeze_axes);
}

std::shared_ptr<ov::Node> create_coordinate_range(const std::shared_ptr<ov::Node>& dim_scalar,
                                                  const std::shared_ptr<ov::Node>& dim_f,
                                                  const GridConstants& consts,
                                                  bool align_corners) {
    if (align_corners) {
        auto is_one = std::make_shared<v1::Equal>(dim_scalar, consts.one);
        auto minus_one = std::make_shared<v1::Subtract>(dim_f, consts.one);
        auto safe_denom = std::make_shared<v1::Select>(is_one, consts.one, to_scalar(minus_one));
        auto step = std::make_shared<v1::Divide>(consts.two, safe_denom);
        auto range_normal = std::make_shared<v4::Range>(consts.minus_one, consts.one_plus_eps, step, ov::element::f32);
        return std::make_shared<v1::Select>(is_one, consts.zero_range, range_normal);
    } else {
        auto step = std::make_shared<v1::Divide>(consts.two, dim_scalar);
        auto half_step = std::make_shared<v1::Multiply>(step, consts.half);
        auto start = std::make_shared<v1::Add>(consts.minus_one, half_step);
        return std::make_shared<v4::Range>(start, consts.one, step, ov::element::f32);
    }
}

std::shared_ptr<ov::Node> construct_original_grid_2d(const ov::Output<ov::Node>& data_size, bool align_corners) {
    GridConstants consts;

    std::vector<std::shared_ptr<ov::Node>> dims, dim_scalars;

    for (int i = 0; i < 2; ++i) {
        auto idx_start = v0::Constant::create(ov::element::i32, Shape{1}, {i});
        auto idx_end = v0::Constant::create(ov::element::i32, Shape{1}, {i + 1});
        auto step = v0::Constant::create(ov::element::i32, Shape{1}, {1});
        auto zero_ax = v0::Constant::create(ov::element::i32, Shape{1}, {0});

        auto dim = std::make_shared<v8::Slice>(data_size, idx_start, idx_end, step, zero_ax);
        auto dim_f = std::make_shared<v0::Convert>(dim, ov::element::f32);
        dims.push_back(dim_f);
        dim_scalars.push_back(to_scalar(dim_f));
    }

    auto range_0 = create_coordinate_range(dim_scalars[0], dims[0], consts, align_corners);
    auto range_1 = create_coordinate_range(dim_scalars[1], dims[1], consts, align_corners);

    auto range_shape_0 = std::make_shared<v3::ShapeOf>(range_0);
    auto range_shape_1 = std::make_shared<v3::ShapeOf>(range_1);
    auto one_i64 = v0::Constant::create(ov::element::i64, Shape{1}, {1});

    auto shape_y = std::make_shared<v0::Concat>(OutputVector{range_shape_0, one_i64}, 0);
    auto shape_x = std::make_shared<v0::Concat>(OutputVector{one_i64, range_shape_1}, 0);
    auto shape_broadcast = std::make_shared<v0::Concat>(OutputVector{range_shape_0, range_shape_1}, 0);

    auto y_reshaped = std::make_shared<v1::Reshape>(range_0, shape_y, false);
    auto x_reshaped = std::make_shared<v1::Reshape>(range_1, shape_x, false);
    auto y_broadcast = std::make_shared<v3::Broadcast>(y_reshaped, shape_broadcast);
    auto x_broadcast = std::make_shared<v3::Broadcast>(x_reshaped, shape_broadcast);

    auto ones = v0::Constant::create(ov::element::f32, Shape{1}, {1.0f});
    auto ones_broadcast = std::make_shared<v3::Broadcast>(ones, shape_broadcast);
    auto axis = v0::Constant::create(ov::element::i32, Shape{1}, {2});

    return std::make_shared<v0::Concat>(OutputVector{std::make_shared<v0::Unsqueeze>(x_broadcast, axis),
                                                     std::make_shared<v0::Unsqueeze>(y_broadcast, axis),
                                                     std::make_shared<v0::Unsqueeze>(ones_broadcast, axis)},
                                        2);
}

std::shared_ptr<ov::Node> construct_original_grid_3d(const ov::Output<ov::Node>& data_size, bool align_corners) {
    const auto f32 = ov::element::f32;
    const auto i32 = ov::element::i32;
    const auto i64 = ov::element::i64;

    GridConstants consts;

    std::vector<std::shared_ptr<ov::Node>> dims, dim_scalars, ranges;

    for (int i = 0; i < 3; ++i) {
        auto idx_start = v0::Constant::create(i32, Shape{1}, {i});
        auto idx_end = v0::Constant::create(i32, Shape{1}, {i + 1});
        auto step = v0::Constant::create(i32, Shape{1}, {1});
        auto zero_ax = v0::Constant::create(i32, Shape{1}, {0});

        auto dim = std::make_shared<v8::Slice>(data_size, idx_start, idx_end, step, zero_ax);
        auto dim_f = std::make_shared<v0::Convert>(dim, f32);
        dims.push_back(dim_f);
        dim_scalars.push_back(to_scalar(dim_f));
        ranges.push_back(create_coordinate_range(dim_scalars[i], dims[i], consts, align_corners));
    }

    std::vector<std::shared_ptr<ov::Node>> range_shapes, broadcast_shapes, grids;
    auto i64_one = v0::Constant::create(i64, Shape{1}, {1});

    for (int i = 0; i < 3; ++i) {
        range_shapes.push_back(std::make_shared<v3::ShapeOf>(ranges[i]));
    }

    for (int i = 0; i < 3; ++i) {
        OutputVector shape_vec(3, i64_one);
        shape_vec[i] = range_shapes[i];
        broadcast_shapes.push_back(std::make_shared<v0::Concat>(shape_vec, 0));
    }

    auto final_shape = std::make_shared<v0::Concat>(OutputVector{range_shapes.begin(), range_shapes.end()}, 0);

    for (int i = 0; i < 3; ++i) {
        auto reshaped = std::make_shared<v1::Reshape>(ranges[i], broadcast_shapes[i], false);
        grids.push_back(std::make_shared<v3::Broadcast>(reshaped, final_shape));
    }

    auto ones = v0::Constant::create(f32, Shape{1}, {1.0f});
    auto ones_b = std::make_shared<v3::Broadcast>(ones, final_shape);
    auto axis = v0::Constant::create(i32, Shape{1}, {3});

    return std::make_shared<v0::Concat>(OutputVector{std::make_shared<v0::Unsqueeze>(grids[2], axis),
                                                     std::make_shared<v0::Unsqueeze>(grids[1], axis),
                                                     std::make_shared<v0::Unsqueeze>(grids[0], axis),
                                                     std::make_shared<v0::Unsqueeze>(ones_b, axis)},
                                        3);
}

std::shared_ptr<ov::Node> apply_affine_transform(const ov::Output<ov::Node>& theta,
                                                 const ov::Output<ov::Node>& grid_homo,
                                                 int spatial_dims) {
    auto tshape = std::make_shared<v3::ShapeOf>(theta);
    auto gshape = std::make_shared<v3::ShapeOf>(grid_homo);

    auto i0 = v0::Constant::create(ov::element::i32, Shape{1}, {0});
    auto i1 = v0::Constant::create(ov::element::i32, Shape{1}, {1});
    auto step = v0::Constant::create(ov::element::i32, Shape{1}, {1});

    auto N = std::make_shared<v8::Slice>(tshape, i0, i1, step, i0);

    std::vector<std::shared_ptr<ov::Node>> spatial_dims_vec;
    for (int i = 0; i < spatial_dims; ++i) {
        auto idx = v0::Constant::create(ov::element::i32, Shape{1}, {i});
        auto idx_next = v0::Constant::create(ov::element::i32, Shape{1}, {i + 1});
        spatial_dims_vec.push_back(std::make_shared<v8::Slice>(gshape, idx, idx_next, step, i0));
    }

    auto spatial_size = spatial_dims_vec[0];
    for (int i = 1; i < spatial_dims; ++i) {
        spatial_size = std::make_shared<v1::Multiply>(spatial_size, spatial_dims_vec[i]);
    }

    auto spatial_size_i64 = std::make_shared<v0::Convert>(spatial_size, ov::element::i64);
    auto coord_dim = v0::Constant::create(ov::element::i64, Shape{1}, {spatial_dims + 1});
    auto resh = std::make_shared<v0::Concat>(OutputVector{spatial_size_i64, coord_dim}, 0);
    auto flat = std::make_shared<v1::Reshape>(grid_homo, resh, false);

    auto perm1 = v0::Constant::create(ov::element::i32, Shape{2}, {1, 0});
    auto trans = std::make_shared<v1::Transpose>(flat, perm1);

    auto mm = std::make_shared<v0::MatMul>(theta, trans);

    auto perm2 = v0::Constant::create(ov::element::i32, Shape{3}, {0, 2, 1});
    auto trans2 = std::make_shared<v1::Transpose>(mm, perm2);

    auto N_i64 = std::make_shared<v0::Convert>(N, ov::element::i64);
    auto output_coord_dim = v0::Constant::create(ov::element::i64, Shape{1}, {spatial_dims});

    OutputVector final_shape_vec = {N_i64};
    for (int i = 0; i < spatial_dims; ++i) {
        final_shape_vec.push_back(std::make_shared<v0::Convert>(spatial_dims_vec[i], ov::element::i64));
    }
    final_shape_vec.push_back(output_coord_dim);

    auto final_shape = std::make_shared<v0::Concat>(final_shape_vec, 0);

    return std::make_shared<v1::Reshape>(trans2, final_shape, false);
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
                                                               v0::Constant::create(ov::element::i32, Shape{1}, {0}));
            auto grid = construct_original_grid_2d(data_size, align_corners);
            return {apply_affine_transform(theta, grid, 2)};
        } else if (rank == 5) {
            const auto data_size = std::make_shared<v8::Slice>(size,
                                                               v0::Constant::create(ov::element::i32, Shape{1}, {2}),
                                                               v0::Constant::create(ov::element::i32, Shape{1}, {5}),
                                                               v0::Constant::create(ov::element::i32, Shape{1}, {1}),
                                                               v0::Constant::create(ov::element::i32, Shape{1}, {0}));
            auto grid = construct_original_grid_3d(data_size, align_corners);
            return {apply_affine_transform(theta, grid, 3)};
        } else {
            FRONT_END_THROW("AffineGrid supports only 4D (2D) or 5D (3D) input sizes.");
        }
    } else {
        FRONT_END_THROW("AffineGrid input 'size' must have a static rank.");
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
