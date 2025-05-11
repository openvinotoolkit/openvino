// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/reshape.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

#include "openvino/frontend/exception.hpp"
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
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/variadic_split.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace reshape {
std::vector<std::size_t> infer_dimensions(const std::string& node_name,
                                          const std::vector<std::size_t>& input_shape,
                                          const std::vector<std::size_t>& output_shape) {
    std::vector<std::size_t> inferred_dims{output_shape};

    // If an output dimension is equal to zero its actual value is copied from the input
    // shape argument.
    for (std::size_t idx = 0; idx < inferred_dims.size(); ++idx) {
        if (inferred_dims.at(idx) == 0) {
            FRONT_END_GENERAL_CHECK(idx < input_shape.size(),
                                    "Node ",
                                    node_name,
                                    " cannot copy dimension from the input data shape because "
                                    "requested index is out of range.");

            inferred_dims.at(idx) = input_shape.at(idx);
        }
    }

    // Check whether there are dimensions equal to -1 in output_shape. There may be at
    // most one such case. Its value is then inferred from the size of the tensor and
    // the remaining dimensions.
    auto neg_value_it = std::find(std::begin(inferred_dims), std::end(inferred_dims), -1);
    if (neg_value_it != std::end(inferred_dims)) {
        // only single '-1' value is allowed
        FRONT_END_GENERAL_CHECK(
            std::find(std::next(neg_value_it), std::end(inferred_dims), -1) == std::end(inferred_dims),
            "Node ",
            node_name,
            " more than one dimension is set to (-1). ",
            "Only one dimension value can be inferred.");

        // Set dimension value to 1 temporarily to be able to calculate its value.
        *neg_value_it = 1;
        std::size_t input_shape_product =
            std::accumulate(std::begin(input_shape), std::end(input_shape), size_t{1}, std::multiplies<std::size_t>());
        std::size_t output_shape_product = std::accumulate(std::begin(inferred_dims),
                                                           std::end(inferred_dims),
                                                           size_t{1},
                                                           std::multiplies<std::size_t>());
        *neg_value_it = input_shape_product / output_shape_product;
    }

    return inferred_dims;
}

ov::Output<ov::Node> interpret_as_scalar(const ov::Output<ov::Node>& node) {
    ov::Shape node_shape = node.get_shape();

    // If node is already a scalar, return original
    if (is_scalar(node_shape)) {
        return node;
    }

    FRONT_END_GENERAL_CHECK((shape_size(node_shape) == 1),
                            "Scalar value can't be derived from a node with ",
                            node_shape);

    // If node is a Constant, recreate as Constant with ov::Shape{}
    if (ov::op::util::is_constant(node.get_node())) {
        const auto value = ov::as_type_ptr<v0::Constant>(node.get_node_shared_ptr())->get_data_ptr();
        return std::make_shared<v0::Constant>(node.get_element_type(), ov::Shape{}, value);
    }

    return ov::op::util::reshape(node, ov::Shape{});
}

ov::Output<ov::Node> reshape_channel_shaped_node_to_nchw(const ov::Output<ov::Node>& node,
                                                         const ov::Output<ov::Node>& expected_rank) {
    // Prepare tail shape (rank = conv.rank - 2): [1, 1, 1, 1, ... ]
    const auto one_const = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    const auto two_const = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    const auto tail_shape_rank = std::make_shared<v1::Subtract>(expected_rank, two_const);
    const auto tail_shape = std::make_shared<v3::Broadcast>(one_const, tail_shape_rank);

    // Construct new bias shape: [1, C, 1, 1, ... ]
    const auto C_dim = std::make_shared<v3::ShapeOf>(node);
    const auto new_shape = std::make_shared<v0::Concat>(ov::OutputVector{one_const, C_dim, tail_shape}, 0);

    return std::make_shared<v1::Reshape>(node, new_shape, false);
}

}  // namespace  reshape
}  // namespace onnx
}  // namespace frontend

namespace op {
namespace util {
std::shared_ptr<ov::Node> reshape(const Output<ov::Node>& value, const Shape& shape) {
    if (value.get_partial_shape().same_scheme(shape)) {
        return value.get_node_shared_ptr();
    } else if (is_scalar(shape)) {
        auto value_rank = value.get_shape().size();
        AxisVector axes_vector(value_rank);
        std::iota(axes_vector.begin(), axes_vector.end(), 0);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, Shape{value_rank}, axes_vector);
        return std::make_shared<ov::op::v0::Squeeze>(value, axes);
    } else {
        auto out_pattern = ov::op::v0::Constant::create(ov::element::i64,
                                                        Shape{shape.size()},
                                                        std::vector<int64_t>(shape.begin(), shape.end()));

        return std::make_shared<ov::op::v1::Reshape>(value, out_pattern, false);
    }
}

std::shared_ptr<ov::Node> reorder_axes(const Output<ov::Node>& value, std::vector<size_t> axes_order) {
    const auto axes_order_const =
        ov::op::v0::Constant::create(ov::element::i64,
                                     Shape{axes_order.size()},
                                     std::vector<int64_t>(axes_order.begin(), axes_order.end()));
    return std::make_shared<ov::op::v1::Transpose>(value, axes_order_const);
}

std::shared_ptr<ov::Node> transpose(const Output<ov::Node>& value) {
    // This part is left to preserve backward compatibility and ensure passing ONNX tests.
    if (value.get_partial_shape().is_static()) {
        std::vector<size_t> axes_order(value.get_shape().size());
        std::iota(begin(axes_order), end(axes_order), 0);
        std::reverse(begin(axes_order), end(axes_order));
        return reorder_axes(value, axes_order);
    }

    const auto input_rank = std::make_shared<ov::op::v0::ShapeOf>(std::make_shared<ov::op::v0::ShapeOf>(value));
    const auto neg_one = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {-1});
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
std::shared_ptr<ov::Node> get_normalized_axis_node(const std::shared_ptr<ov::Node> node_rank, int64_t axis) {
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {axis});
    // shortcut for already positive value
    if (axis >= 0) {
        return axis_node;
    }

    // TODO: What if axis value is beyond acceptable values? [-node_rank,
    // node_rank-1]
    return std::make_shared<ov::op::v1::Add>(node_rank, axis_node);
}
}  // namespace

std::shared_ptr<ov::Node> flatten(const Output<ov::Node>& value, int axis) {
    // First dimension of output tensor is the product of [d_0, ... d_{axis-1}] dimensions of
    // input tensor. The last dimension is the product of the rest of input tensor dimensions:
    // [d_{axis}, ..., d_n]
    std::shared_ptr<ov::Node> output_shape;
    if (axis == 0) {
        output_shape = ov::op::v0::Constant::create(ov::element::i64, Shape{2}, {1, -1});
    } else if (axis == 1) {
        output_shape = ov::op::v0::Constant::create(ov::element::i64, Shape{2}, {0, -1});
    } else {
        const auto value_shape = std::make_shared<ov::op::v0::ShapeOf>(value);
        const auto value_rank = std::make_shared<ov::op::v0::ShapeOf>(value_shape);
        const auto axis_node = get_normalized_axis_node(value_rank, axis);

        const auto first_part_dims =
            std::make_shared<ov::op::v1::StridedSlice>(value_shape,
                                                       ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                                       axis_node,
                                                       std::vector<int64_t>{0},
                                                       std::vector<int64_t>{0});
        const auto first_part_dims_length =
            std::make_shared<ov::op::v1::ReduceProd>(first_part_dims,
                                                     ov::op::v0::Constant::create(ov::element::i64, {}, {0}),
                                                     true);

        const auto remaining_part_length = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        output_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_part_dims_length, remaining_part_length}, 0);
    }
    return std::make_shared<ov::op::v1::Reshape>(value, output_shape, true);
}
}  // namespace util
}  // namespace op
}  // namespace ov
