// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/resize.hpp"

#include <map>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace {
static const std::unordered_set<std::string> supported_modes = {"nearest", "linear", "cubic"};

static const std::unordered_set<std::string> supported_transforms = {"half_pixel",
                                                                     "pytorch_half_pixel",
                                                                     "align_corners",
                                                                     "asymmetric",
                                                                     "tf_half_pixel_for_nn"};

using InterpolateMode = default_opset::Interpolate::InterpolateMode;
static const std::map<std::string, int> interp_mode_map = {{"nearest", static_cast<int>(InterpolateMode::NEAREST)},
                                                           {"linear", static_cast<int>(InterpolateMode::LINEAR_ONNX)},
                                                           {"cubic", static_cast<int>(InterpolateMode::CUBIC)}};

using Transform_mode = default_opset::Interpolate::CoordinateTransformMode;
static const std::map<std::string, int> transform_mode_map = {
    {"half_pixel", static_cast<int>(Transform_mode::HALF_PIXEL)},
    {"pytorch_half_pixel", static_cast<int>(Transform_mode::PYTORCH_HALF_PIXEL)},
    {"align_corners", static_cast<int>(Transform_mode::ALIGN_CORNERS)},
    {"asymmetric", static_cast<int>(Transform_mode::ASYMMETRIC)},
    {"tf_half_pixel_for_nn", static_cast<int>(Transform_mode::TF_HALF_PIXEL_FOR_NN)}};

using Nearest_mode = default_opset::Interpolate::NearestMode;
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

using InterpolateAttrs = default_opset::Interpolate::InterpolateAttrs;

InterpolateAttrs get_resize_attrs(const onnx_import::Node& node) {
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
    attrs.antialias = false;
    attrs.cube_coeff = node.get_attribute_value<float>("cubic_coeff_a", -0.75);
    attrs.pads_begin = {0};
    attrs.pads_end = {0};

    return attrs;
}
}  // namespace

namespace set_11 {
OutputVector resize(const onnx_import::Node& node) {
    // roi input (inputs.at(2)) is ignored because it is used only
    // in "tf_crop_and_resize" which is not handled now
    const auto inputs = node.get_ng_inputs();
    const auto& data = inputs.at(0);

    auto attrs = get_resize_attrs(node);

    if (inputs.size() == 4 && !ngraph::op::is_null(inputs[3])) {
        attrs.shape_calculation_mode = default_opset::Interpolate::ShapeCalcMode::SIZES;
        const auto& sizes = inputs.at(3);
        return {std::make_shared<default_opset::Interpolate>(data, sizes, attrs)};
    } else {
        attrs.shape_calculation_mode = default_opset::Interpolate::ShapeCalcMode::SCALES;
        const auto& scales = inputs.at(2);
        return {std::make_shared<default_opset::Interpolate>(data, scales, attrs)};
    }
}
}  // namespace set_11

namespace set_1 {
OutputVector resize(const onnx_import::Node& node) {
    const auto inputs = node.get_ng_inputs();
    const auto& data = inputs.at(0);
    const auto& scales = inputs.at(1);

    auto attrs = get_resize_attrs(node);
    attrs.shape_calculation_mode = default_opset::Interpolate::ShapeCalcMode::SCALES;

    if (attrs.mode == InterpolateMode::NEAREST) {
        attrs.nearest_mode = Nearest_mode::SIMPLE;
        attrs.coordinate_transformation_mode = Transform_mode::ASYMMETRIC;
    } else if (attrs.mode == InterpolateMode::LINEAR_ONNX) {
        attrs.coordinate_transformation_mode = Transform_mode::ASYMMETRIC;
    }
    return {std::make_shared<default_opset::Interpolate>(data, scales, attrs)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
