// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace detail {
std::shared_ptr<ov::Node> get_zero_point(const ov::OutputVector& inputs) {
    if (inputs.size() == 3 && !ov::op::util::is_null(inputs[2])) {
        const auto& zero_point = inputs[2];

        if (zero_point.get_element_type() != ov::element::f32) {
            return std::make_shared<v0::Convert>(zero_point, ov::element::f32);
        }

        return zero_point.get_node_shared_ptr();
    }
    return nullptr;
}
}  // namespace detail
namespace opset_1 {
ov::OutputVector dequantize_linear(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector inputs{node.get_ov_inputs()};

    FRONT_END_GENERAL_CHECK(2 <= inputs.size() && inputs.size() <= 3,
                            "The DequantizeLinear op expects 2 required and one optional input. Got: ",
                            inputs.size());

    const auto& x = inputs[0];
    const auto& scale = inputs[1];
    const auto zero_point = detail::get_zero_point(inputs);

    common::validate_scalar_input("Dequantization scale", scale.get_node_shared_ptr(), {ov::element::f32});

    const auto converted_x = std::make_shared<v0::Convert>(x, ov::element::f32);

    if (zero_point) {
        common::validate_scalar_input("Zero point", zero_point);
        return {std::make_shared<v1::Multiply>(std::make_shared<v1::Subtract>(converted_x, zero_point), scale)};
    } else {
        return {std::make_shared<v1::Multiply>(converted_x, scale)};
    }
}
ONNX_OP("DequantizeLinear", {1, 12}, ai_onnx::opset_1::dequantize_linear);
}  // namespace opset_1

namespace opset_13 {
namespace detail {
void validate_scale(const ov::Output<ov::Node> scale, const ov::Output<ov::Node> x, const int64_t axis) {
    const auto& scale_shape = scale.get_partial_shape();
    FRONT_END_GENERAL_CHECK(scale_shape.rank().get_length() == 0 || scale_shape.rank().get_length() == 1,
                            "Dequantization scale needs to be a scalar or a vector.");

    if (scale_shape.rank().get_length() == 1) {
        const auto& scale_dim = scale_shape[0];
        const auto& x_shape = x.get_partial_shape();
        const auto& x_dim_at_axis = x_shape[axis];

        FRONT_END_GENERAL_CHECK(scale_dim.compatible(x_dim_at_axis),
                                "The number of dequantization scale elements '",
                                scale_dim,
                                "' must match the input shape dimension '",
                                x_dim_at_axis,
                                " pointed to by the axis attribute: ",
                                axis);
    }
}

void validate_zero_point(const ov::Output<ov::Node> zero_point, const ov::Output<ov::Node> x, const int64_t axis) {
    const auto& zero_point_shape = zero_point.get_partial_shape();
    FRONT_END_GENERAL_CHECK(zero_point_shape.rank().get_length() == 0 || zero_point_shape.rank().get_length() == 1,
                            "Zero point needs to be a scalar or a vector.");

    if (zero_point_shape.rank().get_length() == 1) {
        const auto& zero_point_dim = zero_point_shape[0];
        const auto& x_shape = x.get_partial_shape();
        const auto& x_dim_at_axis = x_shape[axis];

        FRONT_END_GENERAL_CHECK(zero_point_dim.compatible(x_dim_at_axis),
                                "The number of zero point elements '",
                                zero_point_dim,
                                "' must match the input shape dimension '",
                                x_dim_at_axis,
                                " pointed to by the axis attribute: ",
                                axis);
    }
}

std::shared_ptr<ov::Node> reshape_input(const ov::Output<ov::Node>& input,
                                        const int64_t axis,
                                        const ov::PartialShape& x_shape) {
    // these reshapes make sure that dequantization happens over the specified axis
    auto input_rank = input.get_partial_shape().rank();

    // Do not reshape input, if it contains a scalar value
    if (input_rank.is_static() && input_rank.get_length() == 0) {
        return input.get_node_shared_ptr();
    }

    std::vector<int64_t> target_dims;
    for (int64_t i = 0; i < axis; ++i) {
        target_dims.push_back(1);
    }

    // copy dimension at axis from input X
    if (x_shape[axis].is_static()) {
        target_dims.push_back(x_shape[axis].get_length());
    } else {
        target_dims.push_back(-1);
    }

    for (int64_t i = axis + 1; i < x_shape.rank().get_length(); ++i) {
        target_dims.push_back(1);
    }

    const auto target_shape = v0::Constant::create(ov::element::i64, ov::Shape{target_dims.size()}, target_dims);

    return std::make_shared<v1::Reshape>(input, target_shape, true);
}

ov::OutputVector dequantize_linear(const ov::Output<ov::Node>& x,
                                   const ov::Output<ov::Node>& scale,
                                   const std::shared_ptr<ov::Node>& zero_point,
                                   int64_t axis,
                                   const Node& node) {
    const auto& x_shape = x.get_partial_shape();

    FRONT_END_GENERAL_CHECK(x_shape.rank().is_static(), "Rank of the input data tensor has to be known (static).");

    axis = common::normalize_axis(node.get_description(), axis, x_shape.rank());

    validate_scale(scale, x, axis);
    const auto scale_reshaped = reshape_input(scale, axis, x_shape);
    const auto converted_x = std::make_shared<v0::Convert>(x, ov::element::f32);

    if (zero_point) {
        validate_zero_point(zero_point, x, axis);
        return {std::make_shared<v1::Multiply>(
            std::make_shared<v1::Subtract>(converted_x, reshape_input(zero_point, axis, x_shape)),
            scale_reshaped)};
    } else {
        return {std::make_shared<v1::Multiply>(converted_x, scale_reshaped)};
    }
}
}  // namespace detail

ov::OutputVector dequantize_linear(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector inputs{node.get_ov_inputs()};

    FRONT_END_GENERAL_CHECK(2 <= inputs.size() && inputs.size() <= 3,
                            "The DequantizeLinear op expects 2 required and one optional "
                            "input. Got: ",
                            inputs.size());
    const auto& x = inputs[0];
    const auto& scale = inputs[1];
    const auto zero_point = ai_onnx::detail::get_zero_point(inputs);

    const auto& scale_shape = scale.get_partial_shape();
    // per-tensor quantization, axis attribute ignored
    if ((scale_shape.rank().is_static() && scale_shape.size() == 0) ||
        (scale_shape.is_static() && shape_size(scale_shape.get_shape()) == 1)) {
        if (!zero_point) {
            return ai_onnx::opset_1::dequantize_linear(node);
        }
        const auto& zero_point_shape = zero_point->get_output_partial_shape(0);
        if ((zero_point_shape.rank().is_static() && zero_point_shape.size() == 0) ||
            (zero_point_shape.is_static() && shape_size(zero_point_shape.get_shape()) == 1)) {
            return ai_onnx::opset_1::dequantize_linear(node);
        }
    }
    // these reshapes make sure that dequantization happens over the specified axis
    return detail::dequantize_linear(x, scale, zero_point, node.get_attribute_value<int64_t>("axis", 1), node);
}
ONNX_OP("DequantizeLinear", {13, 18}, ai_onnx::opset_13::dequantize_linear);
}  // namespace opset_13

namespace opset_19 {
ONNX_OP("DequantizeLinear", {19, 20}, ai_onnx::opset_13::dequantize_linear);
}  // namespace opset_19

namespace opset_21 {
ov::OutputVector dequantize_linear(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);

    const ov::OutputVector inputs{node.get_ov_inputs()};
    const auto& src_x = inputs[0];
    ov::Output<ov::Node> scale = inputs[1];
    const auto& scale_shape = scale.get_partial_shape();
    ov::Output<ov::Node> zp;

    // When no blocking dequantization is required - use regular DequantizeLinear
    if (scale_shape.rank().is_static() && scale_shape.rank().get_length() <= 1) {
        return ai_onnx::opset_13::dequantize_linear(node);
    }

    FRONT_END_GENERAL_CHECK(scale_shape.rank().is_static(), "Rank of the input data tensor has to be known (static).");
    FRONT_END_GENERAL_CHECK(scale_shape.rank().get_length() == 2,
                            "DequantizeLinear cannot operate with more than 2D scales");
    FRONT_END_GENERAL_CHECK(src_x.get_partial_shape().is_static(),
                            "DequantizeLinear cannot operate with dynamic shapes of input X");

    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
    const auto block_size = static_cast<size_t>(node.get_attribute_value<int64_t>("block_size", 0));

    FRONT_END_GENERAL_CHECK(axis == 0, "Axis != 0 isn't supported");
    FRONT_END_GENERAL_CHECK(block_size > 0, "block_size must be greater than zero");
    FRONT_END_GENERAL_CHECK(
        src_x.get_shape()[0] % block_size == 0,
        "DequantizeLinear doesn't support case when first dimension of X cannot be divided by block_size");

    // For further broadcasting scales and zp - reshape input to a shape [x.shape[0]/block_size, block_size, x.shape[1]]
    ov::Output<ov::Node> broadcastable_x = op::util::reshape(
        src_x,
        Shape{static_cast<size_t>(src_x.get_shape()[0]) / block_size, block_size, src_x.get_shape()[1]});

    const auto& unsqueezed_axes = std::make_shared<v0::Constant>(ov::element::i64, Shape{1}, std::vector<int64_t>{1});

    const auto scale_type = scale.get_element_type();
    if (inputs.size() > 2) {
        zp = inputs[2];
        zp = std::make_shared<v0::Unsqueeze>(zp, unsqueezed_axes);
        if (zp.get_element_type() != scale.get_element_type()) {
            zp = std::make_shared<v0::Convert>(zp, scale_type);
        }
    }

    const auto& x = src_x.get_element_type() == scale_type ? broadcastable_x
                                                           : std::make_shared<v0::Convert>(broadcastable_x, scale_type);

    // Adding additional dimension for broadcasting
    scale = std::make_shared<v0::Unsqueeze>(scale, unsqueezed_axes);

    if (zp.get_node_shared_ptr()) {
        broadcastable_x = std::make_shared<v1::Subtract>(x, zp);
    } else {
        broadcastable_x = x;
    }

    const auto& scaled_x = std::make_shared<v1::Multiply>(broadcastable_x, scale);

    // Returning back a shape
    const auto& reshaped_scaled_x =
        std::make_shared<v1::Reshape>(scaled_x, std::make_shared<v0::ShapeOf>(src_x), false);

    reshaped_scaled_x->set_friendly_name(node.get_name());

    return {reshaped_scaled_x};
}
ONNX_OP("DequantizeLinear", OPSET_SINCE(21), ai_onnx::opset_21::dequantize_linear);
}  // namespace opset_21
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
