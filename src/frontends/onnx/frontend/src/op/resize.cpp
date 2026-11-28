// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"
using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx {
namespace {
static const std::unordered_set<std::string> supported_modes = {"nearest", "linear", "cubic"};

static const std::unordered_set<std::string> supported_transforms = {"half_pixel",
                                                                     "pytorch_half_pixel",
                                                                     "align_corners",
                                                                     "asymmetric",
                                                                     "tf_half_pixel_for_nn"};

using InterpolateMode = v11::Interpolate::InterpolateMode;
static const std::map<std::string, int> interp_mode_map = {{"nearest", static_cast<int>(InterpolateMode::NEAREST)},
                                                           {"linear", static_cast<int>(InterpolateMode::LINEAR_ONNX)},
                                                           {"cubic", static_cast<int>(InterpolateMode::CUBIC)}};

using Transform_mode = v11::Interpolate::CoordinateTransformMode;
static const std::map<std::string, int> transform_mode_map = {
    {"half_pixel", static_cast<int>(Transform_mode::HALF_PIXEL)},
    {"pytorch_half_pixel", static_cast<int>(Transform_mode::PYTORCH_HALF_PIXEL)},
    {"align_corners", static_cast<int>(Transform_mode::ALIGN_CORNERS)},
    {"asymmetric", static_cast<int>(Transform_mode::ASYMMETRIC)},
    {"tf_half_pixel_for_nn", static_cast<int>(Transform_mode::TF_HALF_PIXEL_FOR_NN)}};

using Nearest_mode = v11::Interpolate::NearestMode;
static const std::map<std::string, int> nearest_mode_map = {
    {"round_prefer_floor", static_cast<int>(Nearest_mode::ROUND_PREFER_FLOOR)},
    {"round_prefer_ceil", static_cast<int>(Nearest_mode::ROUND_PREFER_CEIL)},
    {"floor", static_cast<int>(Nearest_mode::FLOOR)},
    {"ceil", static_cast<int>(Nearest_mode::CEIL)}};

static bool is_supported_str_value(const std::unordered_set<std::string>& modes, const std::string& checked_mode) {
    return std::find(modes.begin(), modes.end(), checked_mode) != modes.end();
}

static int mode_as_int(const std::map<std::string, int>& converting_map, const std::string& mode) {
    int result = 0;
    auto it = converting_map.find(mode);
    if (it != converting_map.end()) {
        result = it->second;
    }
    return result;
}

using InterpolateAttrs = v11::Interpolate::InterpolateAttrs;

InterpolateAttrs get_resize_attrs(const ov::frontend::onnx::Node& node) {
    auto get_str_attr = [&node](const std::string& name, const std::string& default_value) {
        return node.get_attribute_value<std::string>(name, default_value);
    };

    const auto mode = get_str_attr("mode", "nearest");
    const auto transform_mode = get_str_attr("coordinate_transformation_mode", "half_pixel");
    const auto nearest_mode = get_str_attr("nearest_mode", "round_prefer_floor");

    if (!is_supported_str_value(supported_modes, mode)) {
        std::string supported_modes_str = "";
        for (const auto& mode_name : supported_modes) {
            supported_modes_str += (mode_name + ", ");
        }
        supported_modes_str.pop_back();
        supported_modes_str.pop_back();
        CHECK_VALID_NODE(node,
                         false,
                         mode,
                         " - this type of interpolation mode is not supported."
                         " Choose one of the following modes: ",
                         supported_modes_str);
    }

    if (!is_supported_str_value(supported_transforms, transform_mode)) {
        std::string supported_modes_str = "";
        for (const auto& mode_name : supported_transforms) {
            supported_modes_str += (mode_name + ", ");
        }
        supported_modes_str.pop_back();
        supported_modes_str.pop_back();
        CHECK_VALID_NODE(node,
                         false,
                         transform_mode,
                         " - this type of coordinate transformation mode is not "
                         "supported. Choose one of the following modes: ",
                         supported_modes_str);
    }

    InterpolateAttrs attrs;
    attrs.mode = static_cast<InterpolateMode>(mode_as_int(interp_mode_map, mode));
    attrs.coordinate_transformation_mode = static_cast<Transform_mode>(mode_as_int(transform_mode_map, transform_mode));
    attrs.nearest_mode = static_cast<Nearest_mode>(mode_as_int(nearest_mode_map, nearest_mode));
    attrs.antialias = (node.get_attribute_value<int64_t>("antialias", 0) == 1);
    attrs.cube_coeff = node.get_attribute_value<float>("cubic_coeff_a", -0.75);
    attrs.pads_begin = {0};
    attrs.pads_end = {0};

    return attrs;
}

ov::Output<ov::Node> f32_scalar(float value) {
    return v0::Constant::create(ov::element::f32, ov::Shape{}, {value});
}

ov::Output<ov::Node> i64_scalar(int64_t value) {
    return v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

// Cubic convolution kernel for a tap at distance d from the sampled position, as in the ONNX reference
// implementation: |d| <= 1 for near taps, 1 < |d| < 2 for far taps.
ov::Output<ov::Node> cubic_kernel(const ov::Output<ov::Node>& d, float a, bool near_tap) {
    if (near_tap) {
        // ((a + 2) * d - (a + 3)) * d * d + 1
        auto poly =
            std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(f32_scalar(a + 2.f), d), f32_scalar(a + 3.f));
        auto sq = std::make_shared<v1::Multiply>(d, d);
        return std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(poly, sq), f32_scalar(1.f));
    }
    // ((a * d - 5a) * d + 8a) * d - 4a
    auto p1 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(f32_scalar(a), d), f32_scalar(5.f * a));
    auto p2 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(p1, d), f32_scalar(8.f * a));
    return std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(p2, d), f32_scalar(4.f * a));
}

// Resizes `data` along one axis with the cubic kernel, dropping the taps outside of [0, in_size - 1] and
// renormalising the weights of the rest. `scale_or_size` is a scalar: a scale or an output size.
ov::Output<ov::Node> resize_axis_cubic_exclude_outside(const ov::Output<ov::Node>& data,
                                                       int64_t axis,
                                                       int64_t rank,
                                                       const ov::Output<ov::Node>& scale_or_size,
                                                       bool is_size,
                                                       Transform_mode transform_mode,
                                                       float cubic_coeff) {
    const auto data_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
    const auto in_size =
        std::make_shared<v0::Convert>(std::make_shared<v8::Gather>(data_shape, i64_scalar(axis), i64_scalar(0)),
                                      ov::element::f32);

    ov::Output<ov::Node> out_size;
    ov::Output<ov::Node> scale;
    if (is_size) {
        out_size = std::make_shared<v0::Convert>(scale_or_size, ov::element::f32);
        scale = std::make_shared<v1::Divide>(out_size, in_size);
    } else {
        scale = std::make_shared<v0::Convert>(scale_or_size, ov::element::f32);
        out_size = std::make_shared<v0::Floor>(std::make_shared<v1::Multiply>(in_size, scale));
    }

    // Output indices -> positions in the input
    const auto out_idx = std::make_shared<v4::Range>(f32_scalar(0.f), out_size, f32_scalar(1.f), ov::element::f32);
    const auto half_pixel = [&]() -> ov::Output<ov::Node> {
        const auto shifted = std::make_shared<v1::Add>(out_idx, f32_scalar(0.5f));
        return std::make_shared<v1::Subtract>(std::make_shared<v1::Divide>(shifted, scale), f32_scalar(0.5f));
    };
    const auto last_in = std::make_shared<v1::Subtract>(in_size, f32_scalar(1.f));
    ov::Output<ov::Node> pos;
    switch (transform_mode) {
    case Transform_mode::HALF_PIXEL:
        pos = half_pixel();
        break;
    case Transform_mode::PYTORCH_HALF_PIXEL: {
        const auto is_single = std::make_shared<v1::Equal>(out_size, f32_scalar(1.f));
        pos = std::make_shared<v1::Select>(is_single, f32_scalar(0.f), half_pixel());
        break;
    }
    case Transform_mode::ASYMMETRIC:
        pos = std::make_shared<v1::Divide>(out_idx, scale);
        break;
    case Transform_mode::ALIGN_CORNERS: {
        const auto is_single = std::make_shared<v1::Equal>(out_size, f32_scalar(1.f));
        const auto last_out = std::make_shared<v1::Subtract>(out_size, f32_scalar(1.f));
        pos = std::make_shared<v1::Select>(
            is_single,
            f32_scalar(0.f),
            std::make_shared<v1::Divide>(std::make_shared<v1::Multiply>(out_idx, last_in), last_out));
        break;
    }
    default:
        OPENVINO_THROW("Unsupported coordinate transformation mode for Resize with exclude_outside=1");
    }

    const auto base = std::make_shared<v0::Floor>(pos);
    const auto frac = std::make_shared<v1::Subtract>(pos, base);
    const auto inv_frac = std::make_shared<v1::Subtract>(f32_scalar(1.f), frac);
    const auto far_frac = std::make_shared<v1::Add>(f32_scalar(1.f), frac);
    const auto far_inv_frac = std::make_shared<v1::Add>(f32_scalar(1.f), inv_frac);
    ov::OutputVector weights = {cubic_kernel(far_frac, cubic_coeff, false),
                                cubic_kernel(frac, cubic_coeff, true),
                                cubic_kernel(inv_frac, cubic_coeff, true),
                                cubic_kernel(far_inv_frac, cubic_coeff, false)};

    ov::OutputVector taps;
    for (int64_t i = 0; i < 4; ++i) {
        const auto tap = std::make_shared<v1::Add>(base, f32_scalar(static_cast<float>(i - 1)));
        const auto inside = std::make_shared<v1::LogicalAnd>(std::make_shared<v1::GreaterEqual>(tap, f32_scalar(0.f)),
                                                             std::make_shared<v1::LessEqual>(tap, last_in));
        weights[i] = std::make_shared<v1::Select>(inside, weights[i], f32_scalar(0.f));
        taps.push_back(tap);
    }
    const auto weights_sum = std::make_shared<v1::Add>(std::make_shared<v1::Add>(weights[0], weights[1]),
                                                       std::make_shared<v1::Add>(weights[2], weights[3]));

    // Weights are broadcast along `axis` of the gathered data
    std::vector<int64_t> bcast_shape(rank, 1);
    bcast_shape[axis] = -1;
    const auto bcast = v0::Constant::create(ov::element::i64, ov::Shape{bcast_shape.size()}, bcast_shape);

    ov::Output<ov::Node> result;
    for (int64_t i = 0; i < 4; ++i) {
        const auto idx = std::make_shared<v0::Convert>(
            std::make_shared<v1::Minimum>(std::make_shared<v1::Maximum>(taps[i], f32_scalar(0.f)), last_in),
            ov::element::i64);
        const auto gathered = std::make_shared<v8::Gather>(data, idx, i64_scalar(axis));
        const auto weight = std::make_shared<v0::Convert>(
            std::make_shared<v1::Reshape>(std::make_shared<v1::Divide>(weights[i], weights_sum), bcast, false),
            data.get_element_type());
        const auto term = std::make_shared<v1::Multiply>(gathered, weight);
        result = i == 0 ? ov::Output<ov::Node>(term) : std::make_shared<v1::Add>(result, term);
    }
    return result;
}

// Interpolate-11 clamps tap indices that fall outside of the input, which is the exclude_outside=0 behaviour of the
// ONNX Resize. For exclude_outside=1 the weights of such taps are set to 0 and the remaining ones are renormalised, so
// cubic resampling is expressed here with OpenVINO operations, separably along each resized axis.
ov::Output<ov::Node> resize_cubic_exclude_outside(const ov::frontend::onnx::Node& node,
                                                  const ov::Output<ov::Node>& data,
                                                  const ov::Output<ov::Node>& scale_or_sizes,
                                                  const InterpolateAttrs& attrs) {
    const auto transform_mode = node.get_attribute_value<std::string>("coordinate_transformation_mode", "half_pixel");
    CHECK_VALID_NODE(node,
                     transform_mode != "tf_half_pixel_for_nn",
                     "exclude_outside=1 is not supported with coordinate_transformation_mode '",
                     transform_mode,
                     "' for cubic mode.");
    CHECK_VALID_NODE(node, !attrs.antialias, "exclude_outside=1 is not supported with antialias=1 for cubic mode.");

    const auto& data_shape = data.get_partial_shape();
    CHECK_VALID_NODE(node, data_shape.rank().is_static(), "exclude_outside=1 requires an input of static rank.");
    CHECK_VALID_NODE(node,
                     data.get_element_type().is_real(),
                     "exclude_outside=1 requires an input of floating point type.");
    const auto rank = data_shape.rank().get_length();

    auto axes = node.get_attribute_value<std::vector<int64_t>>("axes", {});
    if (axes.empty()) {
        axes.resize(rank);
        std::iota(axes.begin(), axes.end(), 0);
    }
    for (auto& axis : axes) {
        axis = common::normalize_axis(node.get_description(), axis, data_shape.rank());
    }

    const bool is_size = attrs.shape_calculation_mode == v11::Interpolate::ShapeCalcMode::SIZES;
    std::vector<double> constant_values;
    if (const auto constant = ov::as_type_ptr<v0::Constant>(scale_or_sizes.get_node_shared_ptr())) {
        if (constant->get_shape() == ov::Shape{axes.size()}) {
            constant_values = constant->cast_vector<double>();
        }
    }

    auto result = data;
    for (size_t i = 0; i < axes.size(); ++i) {
        const auto axis = axes[i];
        // Resizing by a scale of 1 keeps every sample in place
        if (!constant_values.empty()) {
            const auto value = constant_values[i];
            if ((!is_size && value == 1.0) || (is_size && data_shape[axis].is_static() &&
                                               static_cast<double>(data_shape[axis].get_length()) == value)) {
                continue;
            }
        }
        const auto element =
            std::make_shared<v8::Gather>(scale_or_sizes, i64_scalar(static_cast<int64_t>(i)), i64_scalar(0));
        result = resize_axis_cubic_exclude_outside(result,
                                                   axis,
                                                   rank,
                                                   element,
                                                   is_size,
                                                   attrs.coordinate_transformation_mode,
                                                   attrs.cube_coeff);
    }
    return result;
}
}  // namespace

namespace opset_11 {
ov::OutputVector resize(const ov::frontend::onnx::Node& node) {
    // roi input (inputs.at(2)) is ignored because it is used only
    // in "tf_crop_and_resize" which is not handled now
    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs.at(0);

    auto attrs = get_resize_attrs(node);

    ov::Output<ov::Node> scale_or_sizes;
    if (inputs.size() == 4 && !ov::op::util::is_null(inputs[3])) {
        attrs.shape_calculation_mode = v11::Interpolate::ShapeCalcMode::SIZES;
        scale_or_sizes = inputs.at(3);
    } else {
        attrs.shape_calculation_mode = v11::Interpolate::ShapeCalcMode::SCALES;
        scale_or_sizes = inputs.at(2);
    }
    if (attrs.mode == InterpolateMode::CUBIC && node.get_attribute_value<int64_t>("exclude_outside", 0) == 1) {
        return {resize_cubic_exclude_outside(node, data, scale_or_sizes, attrs)};
    }

    auto axes_attr = node.get_attribute_value<std::vector<int64_t>>("axes", {});
    if (axes_attr.empty()) {
        return {std::make_shared<v11::Interpolate>(data, scale_or_sizes, attrs)};
    }

    const auto data_rank = data.get_partial_shape().rank();
    for (auto& axis : axes_attr) {
        axis = common::normalize_axis(node.get_description(), axis, data_rank);
    }

    auto axes = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{axes_attr.size()}, axes_attr);
    return {std::make_shared<v11::Interpolate>(data, scale_or_sizes, axes, attrs)};
}
ONNX_OP("Resize", OPSET_SINCE(11), ai_onnx::opset_11::resize);
}  // namespace opset_11

namespace opset_1 {
ov::OutputVector resize(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs.at(0);
    const auto& scales = inputs.at(1);

    auto attrs = get_resize_attrs(node);
    attrs.shape_calculation_mode = v11::Interpolate::ShapeCalcMode::SCALES;

    if (attrs.mode == InterpolateMode::NEAREST) {
        attrs.nearest_mode = Nearest_mode::SIMPLE;
        attrs.coordinate_transformation_mode = Transform_mode::ASYMMETRIC;
    } else if (attrs.mode == InterpolateMode::LINEAR_ONNX) {
        attrs.coordinate_transformation_mode = Transform_mode::ASYMMETRIC;
    }
    return {std::make_shared<v11::Interpolate>(data, scales, attrs)};
}

ONNX_OP("Resize", OPSET_RANGE(1, 10), ai_onnx::opset_1::resize);
}  // namespace opset_1
}  // namespace ov::frontend::onnx::ai_onnx
