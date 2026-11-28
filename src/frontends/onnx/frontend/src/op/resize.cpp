// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
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

ov::Output<ov::Node> apply_keep_aspect_ratio_policy(const ov::Output<ov::Node>& data,
                                                    const ov::Output<ov::Node>& sizes,
                                                    const std::vector<int64_t>& axes,
                                                    const std::string& policy) {
    ov::Output<ov::Node> input_sizes = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
    if (!axes.empty()) {
        const auto axes_const = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        const auto gather_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        input_sizes = std::make_shared<v8::Gather>(input_sizes, axes_const, gather_axis);
    }
    const auto target_sizes = std::make_shared<v0::Convert>(sizes, ov::element::i64);

    // The spec computes new_size[j] = round_half_up(in[j] * scale), where scale = min/max over i of size[i] / in[i].
    // Rounding is monotonic, so this equals min/max over i of round_half_up(in[j] * size[i] / in[i]), which is
    // computed exactly in integers as (2 * in[j] * size[i] + in[i]) / (2 * in[i]) to avoid floating-point errors.
    const auto two = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    const auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    const auto input_sizes_column = std::make_shared<v0::Unsqueeze>(input_sizes, one);
    const auto doubled_products =
        std::make_shared<v1::Multiply>(std::make_shared<v1::Multiply>(input_sizes_column, two), target_sizes);
    const auto numerators = std::make_shared<v1::Add>(doubled_products, input_sizes);
    const auto denominators = std::make_shared<v1::Multiply>(input_sizes, two);
    const auto candidates = std::make_shared<v1::Divide>(numerators, denominators);

    if (policy == "not_larger") {
        return std::make_shared<v1::ReduceMin>(candidates, one, false);
    }
    return std::make_shared<v1::ReduceMax>(candidates, one, false);
}
}  // namespace

namespace opset_11 {
ov::OutputVector resize(const ov::frontend::onnx::Node& node) {
    // roi input (inputs.at(2)) is ignored because it is used only
    // in "tf_crop_and_resize" which is not handled now
    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs.at(0);

    auto attrs = get_resize_attrs(node);

    const auto keep_aspect_ratio_policy = node.get_attribute_value<std::string>("keep_aspect_ratio_policy", "stretch");
    CHECK_VALID_NODE(node,
                     keep_aspect_ratio_policy == "stretch" || keep_aspect_ratio_policy == "not_larger" ||
                         keep_aspect_ratio_policy == "not_smaller",
                     keep_aspect_ratio_policy,
                     " - this keep_aspect_ratio_policy is not supported."
                     " Choose one of the following policies: stretch, not_larger, not_smaller");

    auto axes_attr = node.get_attribute_value<std::vector<int64_t>>("axes", {});
    const auto data_rank = data.get_partial_shape().rank();
    for (auto& axis : axes_attr) {
        axis = common::normalize_axis(node.get_description(), axis, data_rank);
    }

    ov::Output<ov::Node> scale_or_sizes;
    if (inputs.size() == 4 && !ov::op::util::is_null(inputs[3])) {
        attrs.shape_calculation_mode = v11::Interpolate::ShapeCalcMode::SIZES;
        scale_or_sizes = inputs.at(3);
        // keep_aspect_ratio_policy is only applicable when the output shape is given by "sizes"
        if (keep_aspect_ratio_policy != "stretch") {
            scale_or_sizes = apply_keep_aspect_ratio_policy(data, scale_or_sizes, axes_attr, keep_aspect_ratio_policy);
        }
    } else {
        attrs.shape_calculation_mode = v11::Interpolate::ShapeCalcMode::SCALES;
        scale_or_sizes = inputs.at(2);
    }

    if (axes_attr.empty()) {
        return {std::make_shared<v11::Interpolate>(data, scale_or_sizes, attrs)};
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
