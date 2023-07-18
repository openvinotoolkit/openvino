// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/dequantize_linear.hpp"

#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace detail {
std::shared_ptr<ngraph::Node> get_zero_point(const OutputVector& inputs) {
    if (inputs.size() == 3 && !ngraph::op::is_null(inputs[2])) {
        const auto& zero_point = inputs[2];

        if (zero_point.get_element_type() != element::f32) {
            return std::make_shared<default_opset::Convert>(zero_point, element::f32);
        }

        return zero_point.get_node_shared_ptr();
    }
    return nullptr;
}
}  // namespace detail
namespace set_1 {
OutputVector dequantize_linear(const Node& node) {
    const OutputVector inputs{node.get_ng_inputs()};

    NGRAPH_CHECK(2 <= inputs.size() && inputs.size() <= 3,
                 "The DequantizeLinear op expects 2 required and one optional input. Got: ",
                 inputs.size());

    const auto& x = inputs[0];
    const auto& scale = inputs[1];
    const auto zero_point = detail::get_zero_point(inputs);

    common::validate_scalar_input("Dequantization scale", scale.get_node_shared_ptr(), {element::f32});

    const auto converted_x = std::make_shared<default_opset::Convert>(x, element::f32);

    if (zero_point) {
        common::validate_scalar_input("Zero point", zero_point);
        return {std::make_shared<default_opset::Multiply>(
            std::make_shared<default_opset::Subtract>(converted_x, zero_point),
            scale)};
    } else {
        return {std::make_shared<default_opset::Multiply>(converted_x, scale)};
    }
}
}  // namespace set_1

namespace set_13 {
namespace detail {
void validate_scale(const Output<ngraph::Node> scale, const Output<ngraph::Node> x, const int64_t axis) {
    const auto& scale_shape = scale.get_partial_shape();
    NGRAPH_CHECK(scale_shape.rank().get_length() == 0 || scale_shape.rank().get_length() == 1,
                 "Dequantization scale needs to be a scalar or a vector.");

    if (scale_shape.rank().get_length() == 1) {
        const auto& scale_dim = scale_shape[0];
        const auto& x_shape = x.get_partial_shape();
        const auto& x_dim_at_axis = x_shape[axis];

        NGRAPH_CHECK(scale_dim.compatible(x_dim_at_axis),
                     "The number of dequantization scale elements '",
                     scale_dim,
                     "' must match the input shape dimension '",
                     x_dim_at_axis,
                     " pointed to by the axis attribute: ",
                     axis);
    }
}

void validate_zero_point(const Output<ngraph::Node> zero_point, const Output<ngraph::Node> x, const int64_t axis) {
    const auto& zero_point_shape = zero_point.get_partial_shape();
    NGRAPH_CHECK(zero_point_shape.rank().get_length() == 0 || zero_point_shape.rank().get_length() == 1,
                 "Zero point needs to be a scalar or a vector.");

    if (zero_point_shape.rank().get_length() == 1) {
        const auto& zero_point_dim = zero_point_shape[0];
        const auto& x_shape = x.get_partial_shape();
        const auto& x_dim_at_axis = x_shape[axis];

        NGRAPH_CHECK(zero_point_dim.compatible(x_dim_at_axis),
                     "The number of zero point elements '",
                     zero_point_dim,
                     "' must match the input shape dimension '",
                     x_dim_at_axis,
                     " pointed to by the axis attribute: ",
                     axis);
    }
}

std::shared_ptr<ngraph::Node> reshape_input(const Output<ngraph::Node>& input,
                                            const int64_t axis,
                                            const PartialShape& x_shape) {
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

    const auto target_shape = default_opset::Constant::create(element::i64, Shape{target_dims.size()}, target_dims);

    return std::make_shared<default_opset::Reshape>(input, target_shape, true);
}

OutputVector dequantize_linear(const Output<ngraph::Node>& x,
                               const Output<ngraph::Node>& scale,
                               const std::shared_ptr<ngraph::Node>& zero_point,
                               int64_t axis,
                               const Node& node) {
    const auto& x_shape = x.get_partial_shape();

    NGRAPH_CHECK(x_shape.rank().is_static(), "Rank of the input data tensor has to be known (static).");

    OPENVINO_SUPPRESS_DEPRECATED_START
    axis = ngraph::normalize_axis(node.get_description(), axis, x_shape.rank());
    OPENVINO_SUPPRESS_DEPRECATED_END

    validate_scale(scale, x, axis);
    const auto scale_reshaped = reshape_input(scale, axis, x_shape);
    const auto converted_x = std::make_shared<default_opset::Convert>(x, element::f32);

    if (zero_point) {
        validate_zero_point(zero_point, x, axis);
        return {std::make_shared<default_opset::Multiply>(
            std::make_shared<default_opset::Subtract>(converted_x, reshape_input(zero_point, axis, x_shape)),
            scale_reshaped)};
    } else {
        return {std::make_shared<default_opset::Multiply>(converted_x, scale_reshaped)};
    }
}
}  // namespace detail

OutputVector dequantize_linear(const Node& node) {
    const OutputVector inputs{node.get_ng_inputs()};

    NGRAPH_CHECK(2 <= inputs.size() && inputs.size() <= 3,
                 "The DequantizeLinear op expects 2 required and one optional "
                 "input. Got: ",
                 inputs.size());
    const auto& x = inputs[0];
    const auto& scale = inputs[1];
    const auto zero_point = op::detail::get_zero_point(inputs);

    // per-tensor quantization, axis attribute ignored
    if (scale.get_partial_shape().rank().is_static() && scale.get_partial_shape().rank().get_length() == 0) {
        if (!zero_point || (zero_point->get_output_partial_shape(0).rank().is_static() &&
                            zero_point->get_output_partial_shape(0).rank().get_length() == 0)) {
            return set_1::dequantize_linear(node);
        }
    }
    // these reshapes make sure that dequantization happens over the specified axis
    return detail::dequantize_linear(x, scale, zero_point, node.get_attribute_value<int64_t>("axis", 1), node);
}
}  // namespace set_13
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
