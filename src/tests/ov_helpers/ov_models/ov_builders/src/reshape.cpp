// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/ov_builders/reshape.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace op {
namespace util {
std::shared_ptr<Node> reshape(const Output<Node>& value, const Shape& shape) {
    if (value.get_partial_shape().same_scheme(shape)) {
        return value.get_node_shared_ptr();
    } else if (is_scalar(shape)) {
        auto value_rank = value.get_shape().size();
        AxisVector axes_vector(value_rank);
        std::iota(axes_vector.begin(), axes_vector.end(), 0);
        auto axes = ov::op::v0::Constant::create(element::i64, Shape{value_rank}, axes_vector);
        return std::make_shared<ov::op::v0::Squeeze>(value, axes);
    } else {
        auto out_pattern = ov::op::v0::Constant::create(element::i64,
                                                        Shape{shape.size()},
                                                        std::vector<int64_t>(shape.begin(), shape.end()));

        return std::make_shared<ov::op::v1::Reshape>(value, out_pattern, false);
    }
}

std::shared_ptr<Node> reorder_axes(const Output<Node>& value, std::vector<size_t> axes_order) {
    const auto axes_order_const =
        ov::op::v0::Constant::create(element::i64,
                                     Shape{axes_order.size()},
                                     std::vector<int64_t>(axes_order.begin(), axes_order.end()));
    return std::make_shared<ov::op::v1::Transpose>(value, axes_order_const);
}

std::shared_ptr<Node> transpose(const Output<Node>& value) {
    // This part is left to preserve backward compatibility and ensure passing ONNX tests.
    if (value.get_partial_shape().is_static()) {
        std::vector<size_t> axes_order(value.get_shape().size());
        std::iota(begin(axes_order), end(axes_order), 0);
        std::reverse(begin(axes_order), end(axes_order));
        return reorder_axes(value, axes_order);
    }

    const auto input_rank = std::make_shared<ov::op::v0::ShapeOf>(std::make_shared<ov::op::v0::ShapeOf>(value));
    const auto neg_one = ov::op::v0::Constant::create(element::i64, Shape{}, {-1});
    const auto start_node = std::make_shared<ov::op::v1::Add>(input_rank, neg_one);
    const auto reverse_axes_order = std::make_shared<ov::op::v0::Range>(reshape(start_node, Shape{}),  // start
                                                                        neg_one,   // stop (exclusive)
                                                                        neg_one);  // step
    return std::make_shared<ov::op::v1::Transpose>(value, reverse_axes_order);
}

namespace {
///
/// \brief      Return the node representing normalized axis with respect to
///             provided rank.
///
/// \param[in]  node_rank  The node representing rank used for normalization.
/// \param[in]  axis       The axis value to be normalized.
///
/// \return     The new Constant node representing normalized axis value.
///
std::shared_ptr<Node> get_normalized_axis_node(const std::shared_ptr<Node> node_rank, int64_t axis) {
    auto axis_node = ov::op::v0::Constant::create(element::i64, Shape{1}, {axis});
    // shortcut for already positive value
    if (axis >= 0) {
        return axis_node;
    }

    // TODO: What if axis value is beyond acceptable values? [-node_rank,
    // node_rank-1]
    return std::make_shared<ov::op::v1::Add>(node_rank, axis_node);
}
}  // namespace

std::shared_ptr<Node> flatten(const Output<Node>& value, int axis) {
    // First dimension of output tensor is the product of [d_0, ... d_{axis-1}] dimensions of
    // input tensor. The last dimension is the product of the rest of input tensor dimensions:
    // [d_{axis}, ..., d_n]
    std::shared_ptr<Node> output_shape;
    if (axis == 0) {
        output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, -1});
    } else if (axis == 1) {
        output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {0, -1});
    } else {
        const auto value_shape = std::make_shared<ov::op::v0::ShapeOf>(value);
        const auto value_rank = std::make_shared<ov::op::v0::ShapeOf>(value_shape);
        const auto axis_node = get_normalized_axis_node(value_rank, axis);

        const auto first_part_dims =
            std::make_shared<ov::op::v1::StridedSlice>(value_shape,
                                                       ov::op::v0::Constant::create(element::i64, {1}, {0}),
                                                       axis_node,
                                                       std::vector<int64_t>{0},
                                                       std::vector<int64_t>{0});
        const auto first_part_dims_length =
            std::make_shared<ov::op::v1::ReduceProd>(first_part_dims,
                                                     ov::op::v0::Constant::create(element::i64, {}, {0}),
                                                     true);

        const auto remaining_part_length = ov::op::v0::Constant::create(element::i64, {1}, {-1});

        output_shape =
            std::make_shared<ov::op::v0::Concat>(OutputVector{first_part_dims_length, remaining_part_length}, 0);
    }
    return std::make_shared<ov::op::v1::Reshape>(value, output_shape, true);
}
}  // namespace util
}  // namespace op
}  // namespace ov
