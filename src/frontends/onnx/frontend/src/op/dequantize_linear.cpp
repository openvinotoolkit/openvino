// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"
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
static bool registered =
    register_translator("DequantizeLinear", VersionRange{1, 12}, ai_onnx::opset_1::dequantize_linear);
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

    axis = ov::util::normalize_axis(node.get_description(), axis, x_shape.rank());

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
static bool registered =
    register_translator("DequantizeLinear", VersionRange::since(13), ai_onnx::opset_13::dequantize_linear);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
