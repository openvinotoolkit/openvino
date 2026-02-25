// Copyright (C) 2018-2026 Intel Corporation
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
namespace opset_1 {
namespace detail {
std::shared_ptr<ov::Node> get_zero_point(const ov::OutputVector& inputs) {
    if (inputs.size() == 3 && !ov::op::util::is_null(inputs[2])) {
        const auto& scale = inputs[1];
        const auto& zero_point = inputs[2];

        if (zero_point.get_element_type() != scale.get_element_type()) {
            return std::make_shared<v0::Convert>(zero_point, scale.get_element_type());
        }

        return zero_point.get_node_shared_ptr();
    }
    return nullptr;
}

ov::OutputVector dequantize_linear(const ov::frontend::onnx::Node& node, int64_t opset) {
    const ov::OutputVector inputs{node.get_ov_inputs()};

    const auto& x = inputs[0];
    const auto& scale = inputs[1];
    const auto zero_point = detail::get_zero_point(inputs);

    std::set<ov::element::Type> valid_types = {ov::element::f32};
    if (opset >= 13) {
        valid_types.emplace(ov::element::f16);
        valid_types.emplace(ov::element::bf16);
    }

    common::validate_scalar_input("Dequantization scale", scale.get_node_shared_ptr(), valid_types);

    const auto converted_x = std::make_shared<v0::Convert>(x, scale.get_element_type());

    if (zero_point) {
        common::validate_scalar_input("Zero point", zero_point);
        return {std::make_shared<v1::Multiply>(std::make_shared<v1::Subtract>(converted_x, zero_point), scale)};
    } else {
        return {std::make_shared<v1::Multiply>(converted_x, scale)};
    }
}
}  // namespace detail

ov::OutputVector dequantize_linear(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);

    return detail::dequantize_linear(node, 1);
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
    const auto converted_x = std::make_shared<v0::Convert>(x, scale.get_element_type());

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
    const auto zero_point = ai_onnx::opset_1::detail::get_zero_point(inputs);

    const auto& scale_shape = scale.get_partial_shape();
    // per-tensor quantization, axis attribute ignored
    if ((scale_shape.rank().is_static() && scale_shape.size() == 0) ||
        (scale_shape.is_static() && shape_size(scale_shape.get_shape()) == 1)) {
        if (!zero_point) {
            return ai_onnx::opset_1::detail::dequantize_linear(node, 13);
        }
        const auto& zero_point_shape = zero_point->get_output_partial_shape(0);
        if ((zero_point_shape.rank().is_static() && zero_point_shape.size() == 0) ||
            (zero_point_shape.is_static() && shape_size(zero_point_shape.get_shape()) == 1)) {
            return ai_onnx::opset_1::detail::dequantize_linear(node, 13);
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
    FRONT_END_GENERAL_CHECK(src_x.get_partial_shape().is_static(),
                            "DequantizeLinear cannot operate with dynamic shapes of input X");

    auto axis = node.get_attribute_value<int64_t>("axis", 1);
    const auto block_size = static_cast<size_t>(node.get_attribute_value<int64_t>("block_size", 0));

    FRONT_END_GENERAL_CHECK(block_size > 0, "block_size must be greater than zero");

    // Normalize axis to handle negative values
    axis = common::normalize_axis(node.get_description(), axis, src_x.get_partial_shape().rank());

    // Check that dimension at axis is divisible by block_size
    FRONT_END_GENERAL_CHECK(src_x.get_shape()[axis] % block_size == 0,
                            "DequantizeLinear doesn't support case when dimension of X at axis ",
                            axis,
                            " (",
                            src_x.get_shape()[axis],
                            ") cannot be divided by block_size (",
                            block_size,
                            ")");

    // Check if this is channel-wise quantization (block_size equals dimension size at axis)
    bool is_cw_quantize = (src_x.get_shape()[axis] == block_size);
    if (is_cw_quantize) {
        ov::Output<ov::Node> converted_x = std::make_shared<v0::Convert>(src_x, scale.get_element_type());
        if (inputs.size() > 2) {
            zp = inputs[2];
            zp = std::make_shared<v0::Convert>(zp, scale.get_element_type());
            converted_x = std::make_shared<v1::Subtract>(converted_x, zp);
        }
        auto scaled_x = std::make_shared<v1::Multiply>(converted_x, scale);
        return {scaled_x};
    }

    // Build target shape for blocked quantization
    // For all axes: [..., num_blocks, block_size, ...]
    // - axis=0: [num_blocks, block_size, ...] with block_size at position 1
    // - axis>0: [..., num_blocks, block_size, ...] with block_size at position axis+1
    std::vector<size_t> target_shape_vector;
    const auto& input_shape = src_x.get_partial_shape();
    for (int64_t i = 0; i < input_shape.rank().get_length(); i++) {
        if (i == axis) {
            // Always num_blocks first, then block_size
            target_shape_vector.push_back(static_cast<size_t>(input_shape[i].get_length()) / block_size);
            target_shape_vector.push_back(block_size);
        } else {
            // Other axes remain unchanged
            target_shape_vector.push_back(static_cast<size_t>(input_shape[i].get_length()));
        }
    }
    ov::Output<ov::Node> broadcastable_x = op::util::reshape(src_x, ov::Shape(target_shape_vector));

    // Unsqueeze position for scale/zero_point:
    // block_size is always at position axis+1, so unsqueeze at axis+1
    const auto unsqueeze_axis = axis + 1;
    const auto& unsqueezed_axes =
        std::make_shared<v0::Constant>(ov::element::i64, Shape{1}, std::vector<int64_t>{unsqueeze_axis});

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
