// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_nearest_interpolate_to_gather.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::Matcher;

namespace ov {
namespace pass {
namespace {

using InterpolateBase = ov::op::util::InterpolateBase;
using CoordinateTransformMode = InterpolateBase::CoordinateTransformMode;
using NearestMode = InterpolateBase::NearestMode;

// Mirrors ov::reference::InterpolateEvalHelper::get_input_coords_for_nearest_mode.
float get_source_coordinate(CoordinateTransformMode mode, float x, float scale, int64_t out, int64_t in) {
    switch (mode) {
    case CoordinateTransformMode::PYTORCH_HALF_PIXEL:
        return out > 1 ? (x + 0.5f) / scale - 0.5f : 0.0f;
    case CoordinateTransformMode::ASYMMETRIC:
        return x / scale;
    case CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN:
        return (x + 0.5f) / scale;
    case CoordinateTransformMode::ALIGN_CORNERS:
        return out == 1 ? 0.0f : x * static_cast<float>(in - 1) / static_cast<float>(out - 1);
    case CoordinateTransformMode::HALF_PIXEL:
    default:
        return ((x + 0.5f) / scale) - 0.5f;
    }
}

// Mirrors ov::reference::get_func(NearestMode).
int64_t get_nearest_pixel(NearestMode mode, float x, bool is_downsample) {
    switch (mode) {
    case NearestMode::ROUND_PREFER_CEIL:
        return static_cast<int64_t>(std::round(x));
    case NearestMode::FLOOR:
        return static_cast<int64_t>(std::floor(x));
    case NearestMode::CEIL:
        return static_cast<int64_t>(std::ceil(x));
    case NearestMode::SIMPLE:
        return is_downsample ? static_cast<int64_t>(std::ceil(x)) : static_cast<int64_t>(x);
    case NearestMode::ROUND_PREFER_FLOOR:
    default:
        if (x == static_cast<int64_t>(x) + 0.5f) {
            return static_cast<int64_t>(std::floor(x));
        }
        return static_cast<int64_t>(std::round(x));
    }
}

bool has_non_zero_padding(const std::vector<size_t>& pads) {
    return std::any_of(pads.begin(), pads.end(), [](size_t pad) {
        return pad != 0;
    });
}

}  // namespace

ConvertNearestInterpolateToGather::ConvertNearestInterpolateToGather() {
    MATCHER_SCOPE(ConvertNearestInterpolateToGather);

    const auto interpolate_pattern = ov::pass::pattern::wrap_type<ov::op::v11::Interpolate, ov::op::v4::Interpolate>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto node_handle = m.get_match_root();
        const auto node = std::dynamic_pointer_cast<InterpolateBase>(node_handle);
        if (!node || transformation_callback(node)) {
            return false;
        }

        const auto& attrs = node->get_attrs();
        if (attrs.mode != InterpolateBase::InterpolateMode::NEAREST || attrs.antialias ||
            has_non_zero_padding(attrs.pads_begin) || has_non_zero_padding(attrs.pads_end)) {
            return false;
        }

        const auto& input_pshape = node->get_input_partial_shape(0);
        const auto& output_pshape = node->get_output_partial_shape(0);
        if (input_pshape.is_dynamic() || output_pshape.is_dynamic()) {
            return false;
        }
        const auto input_shape = input_pshape.to_shape();
        const auto output_shape = output_pshape.to_shape();
        const size_t rank = input_shape.size();
        if (rank < 2 || output_shape.size() != rank) {
            return false;
        }

        std::vector<int64_t> axes;
        if (node->get_input_size() > 2) {
            const auto axes_constant = ov::util::get_constant_from_source(node->input_value(2));
            if (!axes_constant) {
                return false;
            }
            axes = axes_constant->cast_vector<int64_t>();
        }
        if (axes.empty()) {
            axes.resize(rank);
            std::iota(axes.begin(), axes.end(), int64_t{0});
        }

        const auto scales_or_sizes = ov::util::get_constant_from_source(node->input_value(1));
        if (!scales_or_sizes) {
            return false;
        }
        const auto is_scales_mode = attrs.shape_calculation_mode == InterpolateBase::ShapeCalcMode::SCALES;
        const auto scales = is_scales_mode ? scales_or_sizes->cast_vector<float>() : std::vector<float>{};
        const auto sizes = is_scales_mode ? std::vector<int64_t>{} : scales_or_sizes->cast_vector<int64_t>();
        const auto axis_values_count = is_scales_mode ? scales.size() : sizes.size();
        if (axis_values_count != axes.size()) {
            return false;
        }

        ov::pass::NodeRegistry registry;
        ov::Output<ov::Node> result = node->input_value(0);
        bool rewritten = false;

        for (size_t i = 0; i < axes.size(); ++i) {
            const auto axis = axes[i] < 0 ? axes[i] + static_cast<int64_t>(rank) : axes[i];
            if (axis < 0 || static_cast<size_t>(axis) >= rank) {
                return false;
            }
            const auto in = input_shape[axis];
            const auto out = output_shape[axis];
            if (in == out) {
                continue;
            }
            const auto scale = is_scales_mode ? scales[i] : static_cast<float>(out) / static_cast<float>(in);
            if (!(scale > 1.0f)) {
                continue;
            }

            std::vector<int64_t> indices(out);
            for (int64_t out_idx = 0; out_idx < static_cast<int64_t>(out); ++out_idx) {
                const auto source = get_source_coordinate(attrs.coordinate_transformation_mode,
                                                          static_cast<float>(out_idx),
                                                          scale,
                                                          static_cast<int64_t>(out),
                                                          static_cast<int64_t>(in));
                const auto nearest = get_nearest_pixel(attrs.nearest_mode, source, scale < 1.0f);
                indices[out_idx] = std::max(int64_t{0}, std::min(nearest, static_cast<int64_t>(in) - 1));
            }

            const auto indices_constant =
                registry.make<ov::op::v0::Constant>(ov::element::i64, ov::Shape{out}, indices);
            const auto axis_constant = registry.make<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, axis);
            result = registry.make<ov::op::v8::Gather>(result, indices_constant, axis_constant);
            rewritten = true;
        }

        if (!rewritten) {
            return false;
        }

        result.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node_handle, registry.get());
        replace_node(node_handle, result.get_node_shared_ptr());
        return true;
    };

    auto matcher = std::make_shared<Matcher>(interpolate_pattern, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace pass
}  // namespace ov
