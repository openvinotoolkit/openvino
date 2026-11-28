// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"
using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx {
namespace {
static const std::unordered_set<std::string> supported_modes = {"nearest", "linear", "cubic"};

static const std::unordered_set<std::string> supported_transforms =
    {"half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric", "tf_half_pixel_for_nn", "half_pixel_symmetric"};

using InterpolateMode = v11::Interpolate::InterpolateMode;
static const std::map<std::string, int> interp_mode_map = {{"nearest", static_cast<int>(InterpolateMode::NEAREST)},
                                                           {"linear", static_cast<int>(InterpolateMode::LINEAR_ONNX)},
                                                           {"cubic", static_cast<int>(InterpolateMode::CUBIC)}};

using InterpolateTransformMode = v11::Interpolate::CoordinateTransformMode;
enum class Transform_mode {
    HALF_PIXEL = static_cast<int>(InterpolateTransformMode::HALF_PIXEL),
    PYTORCH_HALF_PIXEL = static_cast<int>(InterpolateTransformMode::PYTORCH_HALF_PIXEL),
    ALIGN_CORNERS = static_cast<int>(InterpolateTransformMode::ALIGN_CORNERS),
    ASYMMETRIC = static_cast<int>(InterpolateTransformMode::ASYMMETRIC),
    TF_HALF_PIXEL_FOR_NN = static_cast<int>(InterpolateTransformMode::TF_HALF_PIXEL_FOR_NN),
    HALF_PIXEL_SYMMETRIC = 5
};
static const std::map<std::string, int> transform_mode_map = {
    {"half_pixel", static_cast<int>(Transform_mode::HALF_PIXEL)},
    {"pytorch_half_pixel", static_cast<int>(Transform_mode::PYTORCH_HALF_PIXEL)},
    {"align_corners", static_cast<int>(Transform_mode::ALIGN_CORNERS)},
    {"asymmetric", static_cast<int>(Transform_mode::ASYMMETRIC)},
    {"tf_half_pixel_for_nn", static_cast<int>(Transform_mode::TF_HALF_PIXEL_FOR_NN)},
    {"half_pixel_symmetric", static_cast<int>(Transform_mode::HALF_PIXEL_SYMMETRIC)}};

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
    const auto parsed_transform_mode = mode_as_int(transform_mode_map, transform_mode);
    // Core Interpolate has no HALF_PIXEL_SYMMETRIC enum; its frontend decomposition handles this mode.
    attrs.coordinate_transformation_mode =
        parsed_transform_mode == static_cast<int>(Transform_mode::HALF_PIXEL_SYMMETRIC)
            ? InterpolateTransformMode::HALF_PIXEL
            : static_cast<InterpolateTransformMode>(parsed_transform_mode);
    attrs.nearest_mode = static_cast<Nearest_mode>(mode_as_int(nearest_mode_map, nearest_mode));
    attrs.antialias = (node.get_attribute_value<int64_t>("antialias", 0) == 1);
    attrs.cube_coeff = node.get_attribute_value<float>("cubic_coeff_a", -0.75);
    attrs.pads_begin = {0};
    attrs.pads_end = {0};

    return attrs;
}

ov::Output<ov::Node> compute_source_coords(const ov::Output<ov::Node>& out_indices_f32,
                                           const ov::Output<ov::Node>& in_dim_f32,
                                           const ov::Output<ov::Node>& out_dim_f32,
                                           const ov::Output<ov::Node>& scale_f32,
                                           Transform_mode transform_mode) {
    const auto zero_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    const auto half_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.5f});
    const auto one_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});

    switch (transform_mode) {
    case Transform_mode::HALF_PIXEL_SYMMETRIC: {
        const auto two_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
        auto out_to_in = std::make_shared<v1::Divide>(out_dim_f32, in_dim_f32);
        auto adjustment = std::make_shared<v1::Maximum>(out_to_in, one_f);
        auto adjusted_out_dim = std::make_shared<v1::Divide>(out_dim_f32, adjustment);
        auto offset = std::make_shared<v1::Divide>(std::make_shared<v1::Subtract>(in_dim_f32, adjusted_out_dim), two_f);
        auto shifted = std::make_shared<v1::Add>(out_indices_f32, half_f);
        auto centered = std::make_shared<v1::Subtract>(std::make_shared<v1::Divide>(shifted, adjustment), half_f);
        return std::make_shared<v1::Add>(centered, offset);
    }
    case Transform_mode::HALF_PIXEL: {
        auto shifted = std::make_shared<v1::Add>(out_indices_f32, half_f);
        auto scaled = std::make_shared<v1::Divide>(shifted, scale_f32);
        return std::make_shared<v1::Subtract>(scaled, half_f);
    }
    case Transform_mode::PYTORCH_HALF_PIXEL: {
        auto shifted = std::make_shared<v1::Add>(out_indices_f32, half_f);
        auto scaled = std::make_shared<v1::Divide>(shifted, scale_f32);
        auto half_pixel_res = std::make_shared<v1::Subtract>(scaled, half_f);
        auto is_one = std::make_shared<v1::Equal>(out_dim_f32, one_f);
        return std::make_shared<v1::Select>(is_one, zero_f, half_pixel_res);
    }
    case Transform_mode::ALIGN_CORNERS: {
        auto out_minus_1 = std::make_shared<v1::Subtract>(out_dim_f32, one_f);
        auto in_minus_1 = std::make_shared<v1::Subtract>(in_dim_f32, one_f);
        auto is_one = std::make_shared<v1::Equal>(out_dim_f32, one_f);
        auto safe_denom = std::make_shared<v1::Select>(is_one, one_f, out_minus_1);
        auto ratio = std::make_shared<v1::Divide>(in_minus_1, safe_denom);
        auto res = std::make_shared<v1::Multiply>(out_indices_f32, ratio);
        return std::make_shared<v1::Select>(is_one, zero_f, res);
    }
    case Transform_mode::TF_HALF_PIXEL_FOR_NN: {
        auto shifted = std::make_shared<v1::Add>(out_indices_f32, half_f);
        return std::make_shared<v1::Divide>(shifted, scale_f32);
    }
    case Transform_mode::ASYMMETRIC:
    default: {
        return std::make_shared<v1::Divide>(out_indices_f32, scale_f32);
    }
    }
}

ov::Output<ov::Node> compute_cubic_weight(const ov::Output<ov::Node>& dist, float a) {
    const auto zero_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    const auto one_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    const auto two_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
    const auto a_plus_2 = v0::Constant::create(ov::element::f32, ov::Shape{}, {a + 2.0f});
    const auto a_plus_3 = v0::Constant::create(ov::element::f32, ov::Shape{}, {a + 3.0f});
    const auto a_const = v0::Constant::create(ov::element::f32, ov::Shape{}, {a});
    const auto five_a = v0::Constant::create(ov::element::f32, ov::Shape{}, {5.0f * a});
    const auto eight_a = v0::Constant::create(ov::element::f32, ov::Shape{}, {8.0f * a});
    const auto four_a = v0::Constant::create(ov::element::f32, ov::Shape{}, {4.0f * a});

    auto d2 = std::make_shared<v1::Multiply>(dist, dist);
    auto d3 = std::make_shared<v1::Multiply>(d2, dist);

    auto term1 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(a_plus_2, dist), a_plus_3);
    auto w_le_1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(term1, d2), one_f);

    auto t3 = std::make_shared<v1::Multiply>(a_const, d3);
    auto t2 = std::make_shared<v1::Multiply>(five_a, d2);
    auto t1 = std::make_shared<v1::Multiply>(eight_a, dist);
    auto w_lt_2 =
        std::make_shared<v1::Subtract>(std::make_shared<v1::Add>(std::make_shared<v1::Subtract>(t3, t2), t1), four_a);

    auto le_1 = std::make_shared<v1::LessEqual>(dist, one_f);
    auto lt_2 = std::make_shared<v1::Less>(dist, two_f);
    auto inner = std::make_shared<v1::Select>(le_1, w_le_1, w_lt_2);
    return std::make_shared<v1::Select>(lt_2, inner, zero_f);
}

ov::Output<ov::Node> resize_1d_exclude_outside(const ov::Output<ov::Node>& data,
                                               int64_t axis,
                                               size_t data_rank,
                                               const ov::Output<ov::Node>& in_dim_i64,
                                               const ov::Output<ov::Node>& out_dim_i64,
                                               const ov::Output<ov::Node>& scale_f32,
                                               const InterpolateAttrs& attrs) {
    const auto orig_type = data.get_element_type();
    ov::Output<ov::Node> work_data = data;
    if (orig_type != ov::element::f32) {
        work_data = std::make_shared<v0::Convert>(data, ov::element::f32);
    }

    const auto zero_i64 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto one_i64 = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto zero_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    const auto one_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});

    auto out_indices_i64 = std::make_shared<v4::Range>(zero_i64, out_dim_i64, one_i64, ov::element::i64);
    auto out_indices_f32 = std::make_shared<v0::Convert>(out_indices_i64, ov::element::f32);
    auto in_dim_f32 = std::make_shared<v0::Convert>(in_dim_i64, ov::element::f32);
    auto out_dim_f32 = std::make_shared<v0::Convert>(out_dim_i64, ov::element::f32);

    auto x = compute_source_coords(out_indices_f32, in_dim_f32, out_dim_f32, scale_f32, transform_mode);

    std::vector<ov::Output<ov::Node>> tap_indices_i64;
    std::vector<ov::Output<ov::Node>> tap_weights_f32;

    if (attrs.mode == InterpolateMode::CUBIC) {
        auto x0_f = std::make_shared<v0::Floor>(x);
        auto x0_i64 = std::make_shared<v0::Convert>(x0_f, ov::element::i64);
        auto s = std::make_shared<v1::Subtract>(x, x0_f);

        for (int64_t k = -1; k <= 2; ++k) {
            auto k_i64 = v0::Constant::create(ov::element::i64, ov::Shape{}, {k});
            auto k_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {static_cast<float>(k)});
            auto tap_idx = std::make_shared<v1::Add>(x0_i64, k_i64);
            auto dist = std::make_shared<v0::Abs>(std::make_shared<v1::Subtract>(s, k_f));
            auto w = compute_cubic_weight(dist, attrs.cube_coeff);
            tap_indices_i64.push_back(tap_idx);
            tap_weights_f32.push_back(w);
        }
    } else if (attrs.mode == InterpolateMode::LINEAR_ONNX) {
        auto x0_f = std::make_shared<v0::Floor>(x);
        auto x0_i64 = std::make_shared<v0::Convert>(x0_f, ov::element::i64);
        auto s = std::make_shared<v1::Subtract>(x, x0_f);

        tap_indices_i64.push_back(x0_i64);
        tap_indices_i64.push_back(std::make_shared<v1::Add>(x0_i64, one_i64));
        tap_weights_f32.push_back(std::make_shared<v1::Subtract>(one_f, s));
        tap_weights_f32.push_back(s);
    } else {
        ov::Output<ov::Node> nearest_f;
        const auto half_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.5f});
        switch (attrs.nearest_mode) {
        case Nearest_mode::FLOOR:
            nearest_f = std::make_shared<v0::Floor>(x);
            break;
        case Nearest_mode::CEIL: {
            auto neg_x = std::make_shared<v1::Subtract>(zero_f, x);
            nearest_f = std::make_shared<v1::Subtract>(zero_f, std::make_shared<v0::Floor>(neg_x));
            break;
        }
        case Nearest_mode::ROUND_PREFER_CEIL:
            nearest_f = std::make_shared<v0::Floor>(std::make_shared<v1::Add>(x, half_f));
            break;
        case Nearest_mode::ROUND_PREFER_FLOOR:
        default:
            nearest_f = std::make_shared<v5::Round>(x, v5::Round::RoundMode::HALF_TO_EVEN);
            break;
        }
        tap_indices_i64.push_back(std::make_shared<v0::Convert>(nearest_f, ov::element::i64));
        tap_weights_f32.push_back(one_f);
    }

    const size_t num_taps = tap_indices_i64.size();
    std::vector<ov::Output<ov::Node>> masked_weights;
    masked_weights.reserve(num_taps);

    ov::Output<ov::Node> weight_sum = zero_f;
    for (size_t i = 0; i < num_taps; ++i) {
        auto ge_zero = std::make_shared<v1::GreaterEqual>(tap_indices_i64[i], zero_i64);
        auto lt_in_dim = std::make_shared<v1::Less>(tap_indices_i64[i], in_dim_i64);
        auto in_bounds = std::make_shared<v1::LogicalAnd>(ge_zero, lt_in_dim);
        auto masked_w = std::make_shared<v1::Select>(in_bounds, tap_weights_f32[i], zero_f);
        masked_weights.push_back(masked_w);
        weight_sum = std::make_shared<v1::Add>(weight_sum, masked_w);
    }

    auto sum_is_zero = std::make_shared<v1::Equal>(weight_sum, zero_f);
    auto safe_sum = std::make_shared<v1::Select>(sum_is_zero, one_f, weight_sum);

    std::vector<int64_t> weight_shape_vals(data_rank, 1);
    weight_shape_vals[static_cast<size_t>(axis)] = -1;
    auto weight_target_shape =
        std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{data_rank}, weight_shape_vals);

    auto max_valid_idx = std::make_shared<v1::Subtract>(in_dim_i64, one_i64);
    auto gather_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});

    ov::Output<ov::Node> accum;
    for (size_t i = 0; i < num_taps; ++i) {
        auto norm_w = std::make_shared<v1::Select>(sum_is_zero,
                                                   zero_f,
                                                   std::make_shared<v1::Divide>(masked_weights[i], safe_sum));
        auto w_broadcast = std::make_shared<v1::Reshape>(norm_w, weight_target_shape, false);

        auto clamped_idx =
            std::make_shared<v1::Minimum>(std::make_shared<v1::Maximum>(tap_indices_i64[i], zero_i64), max_valid_idx);
        auto gathered = std::make_shared<v8::Gather>(work_data, clamped_idx, gather_axis);
        auto weighted = std::make_shared<v1::Multiply>(gathered, w_broadcast);

        if (i == 0) {
            accum = weighted;
        } else {
            accum = std::make_shared<v1::Add>(accum, weighted);
        }
    }

    if (orig_type != ov::element::f32) {
        return std::make_shared<v0::Convert>(accum, orig_type);
    }
    return accum;
}

ov::OutputVector resize_with_exclude_outside(const ov::frontend::onnx::Node& node,
                                             const ov::Output<ov::Node>& data,
                                             const ov::Output<ov::Node>& scale_or_sizes,
                                             const std::vector<int64_t>& axes_attr,
                                             const InterpolateAttrs& attrs,
                                             Transform_mode transform_mode) {
    const auto data_pshape = data.get_partial_shape();
    CHECK_VALID_NODE(node, data_pshape.rank().is_static(), "Resize with exclude_outside=1 requires static input rank.");
    const size_t data_rank = data_pshape.rank().get_length();

    std::vector<int64_t> target_axes;
    if (axes_attr.empty()) {
        target_axes.resize(data_rank);
        std::iota(target_axes.begin(), target_axes.end(), 0);
    } else {
        target_axes = axes_attr;
    }

    auto data_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
    const auto zero_i64 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

    ov::Output<ov::Node> current = data;
    for (size_t i = 0; i < target_axes.size(); ++i) {
        const int64_t axis = target_axes[i];
        auto axis_scalar = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        auto idx_scalar = v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(i)});

        auto in_dim_i64 = std::make_shared<v8::Gather>(data_shape, axis_scalar, zero_i64);
        ov::Output<ov::Node> out_dim_i64;
        ov::Output<ov::Node> scale_f32;

        if (attrs.shape_calculation_mode == v11::Interpolate::ShapeCalcMode::SIZES) {
            auto size_elem = std::make_shared<v8::Gather>(scale_or_sizes, idx_scalar, zero_i64);
            out_dim_i64 = std::make_shared<v0::Convert>(size_elem, ov::element::i64);
            auto out_dim_f = std::make_shared<v0::Convert>(out_dim_i64, ov::element::f32);
            auto in_dim_f = std::make_shared<v0::Convert>(in_dim_i64, ov::element::f32);
            scale_f32 = std::make_shared<v1::Divide>(out_dim_f, in_dim_f);
        } else {
            auto scale_elem = std::make_shared<v8::Gather>(scale_or_sizes, idx_scalar, zero_i64);
            scale_f32 = std::make_shared<v0::Convert>(scale_elem, ov::element::f32);
            auto in_dim_f = std::make_shared<v0::Convert>(in_dim_i64, ov::element::f32);
            auto scaled_f = std::make_shared<v0::Floor>(std::make_shared<v1::Multiply>(in_dim_f, scale_f32));
            out_dim_i64 = std::make_shared<v0::Convert>(scaled_f, ov::element::i64);
        }

        if (data_pshape[axis].is_static()) {
            const int64_t static_in_dim = data_pshape[axis].get_length();
            if (const auto const_node = ov::as_type_ptr<v0::Constant>(scale_or_sizes.get_node_shared_ptr())) {
                if (attrs.shape_calculation_mode == v11::Interpolate::ShapeCalcMode::SIZES) {
                    const auto sizes_vec = const_node->cast_vector<int64_t>();
                    if (i < sizes_vec.size() && sizes_vec[i] == static_in_dim) {
                        continue;
                    }
                } else {
                    const auto scales_vec = const_node->cast_vector<float>();
                    if (i < scales_vec.size() && scales_vec[i] == 1.0f) {
                        continue;
                    }
                }
            }
        }

        current = resize_1d_exclude_outside(current, axis, data_rank, in_dim_i64, out_dim_i64, scale_f32, attrs);
    }

    return {current};
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
    auto axes_attr = node.get_attribute_value<std::vector<int64_t>>("axes", {});
    for (auto& axis : axes_attr) {
        axis = common::normalize_axis(node.get_description(), axis, data.get_partial_shape().rank());
    }

    if (node.get_attribute_value<int64_t>("exclude_outside", 0) == 1) {
        const auto transform_mode = static_cast<Transform_mode>(
            mode_as_int(transform_mode_map,
                        node.get_attribute_value<std::string>("coordinate_transformation_mode", "half_pixel")));
        return resize_with_exclude_outside(node, data, scale_or_sizes, axes_attr, attrs, transform_mode);
    }
    if (node.get_attribute_value<std::string>("coordinate_transformation_mode", "half_pixel") ==
        "half_pixel_symmetric") {
        CHECK_VALID_NODE(node,
                         false,
                         "half_pixel_symmetric",
                         " requires the frontend Resize decomposition and is only supported here with "
                         "exclude_outside=1.");
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
        attrs.coordinate_transformation_mode = InterpolateTransformMode::ASYMMETRIC;
    } else if (attrs.mode == InterpolateMode::LINEAR_ONNX) {
        attrs.coordinate_transformation_mode = InterpolateTransformMode::ASYMMETRIC;
    }
    return {std::make_shared<v11::Interpolate>(data, scales, attrs)};
}

ONNX_OP("Resize", OPSET_RANGE(1, 10), ai_onnx::opset_1::resize);
}  // namespace opset_1
}  // namespace ov::frontend::onnx::ai_onnx
