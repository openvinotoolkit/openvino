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
#include "openvino/core/except.hpp"
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

// All conditions that make the decomposition applicable and cheap to precompute.
bool is_nearest_static_interpolate(const ov::Output<ov::Node>& output) {
    const auto node = std::dynamic_pointer_cast<InterpolateBase>(output.get_node_shared_ptr());
    if (!node) {
        return false;
    }
    const auto& attrs = node->get_attrs();
    const auto& output_pshape = output.get_partial_shape();
    const auto& input_pshape = node->get_input_partial_shape(0);
    return attrs.mode == InterpolateBase::InterpolateMode::NEAREST && !attrs.antialias &&
           !has_non_zero_padding(attrs.pads_begin) && !has_non_zero_padding(attrs.pads_end) &&
           output_pshape.is_static() && output_pshape.rank().get_length() >= 2 && input_pshape.is_static() &&
           input_pshape.rank() == output_pshape.rank();
}

}  // namespace

ConvertNearestInterpolateToGather::ConvertNearestInterpolateToGather() {
    MATCHER_SCOPE(ConvertNearestInterpolateToGather);

    const auto interpolate_pattern =
        ov::pass::pattern::wrap_type<ov::op::v11::Interpolate, ov::op::v4::Interpolate>(is_nearest_static_interpolate);

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto node_handle = m.get_pattern_value_map().at(interpolate_pattern).get_node_shared_ptr();
        const auto node = std::dynamic_pointer_cast<InterpolateBase>(node_handle);
        OPENVINO_ASSERT(node, "ConvertNearestInterpolateToGather: matched node is expected to be Interpolate-4/11");

        if (transformation_callback(node_handle)) {
            return false;
        }

        const auto& attrs = node->get_attrs();
        const auto input_shape = node->get_input_partial_shape(0).to_shape();
        const auto output_shape = node->get_output_partial_shape(0).to_shape();
        const size_t rank = input_shape.size();
        const auto is_scales_mode = attrs.shape_calculation_mode == InterpolateBase::ShapeCalcMode::SCALES;

        // v11 has a single `scales_or_sizes` input (port 1) and an optional `axes` input (port 2); v4 keeps the
        // legacy layout: `output_shape` (port 1), `scales` (port 2) and an optional `axes` input (port 3).
        const auto is_v11 = std::dynamic_pointer_cast<ov::op::v11::Interpolate>(node_handle) != nullptr;
        OPENVINO_ASSERT(is_v11 || std::dynamic_pointer_cast<ov::op::v4::Interpolate>(node_handle),
                        "ConvertNearestInterpolateToGather: matched node is expected to be Interpolate-4/11");
        const size_t values_port = is_v11 ? 1 : (is_scales_mode ? 2 : 1);
        const size_t axes_port = is_v11 ? 2 : 3;

        std::vector<int64_t> axes;
        if (node->get_input_size() > axes_port) {
            const auto axes_constant = ov::util::get_constant_from_source(node->input_value(axes_port));
            if (!axes_constant) {
                return false;
            }
            axes = axes_constant->cast_vector<int64_t>();
        }
        if (axes.empty()) {
            axes.resize(rank);
            std::iota(axes.begin(), axes.end(), int64_t{0});
        }

        const auto scales_or_sizes = ov::util::get_constant_from_source(node->input_value(values_port));
        if (!scales_or_sizes) {
            return false;
        }
        const auto scales = is_scales_mode ? scales_or_sizes->cast_vector<float>() : std::vector<float>{};
        const auto sizes = is_scales_mode ? std::vector<int64_t>{} : scales_or_sizes->cast_vector<int64_t>();
        if ((is_scales_mode ? scales.size() : sizes.size()) != axes.size()) {
            return false;
        }

        ov::pass::NodeRegistry registry;
        ov::Output<ov::Node> result = node->input_value(0);
        bool rewritten = false;

        for (size_t i = 0; i < axes.size(); ++i) {
            const auto axis = axes[i] < 0 ? axes[i] + static_cast<int64_t>(rank) : axes[i];
            OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                            "ConvertNearestInterpolateToGather: axis is out of range");
            const auto in = input_shape[axis];
            const auto out = output_shape[axis];
            if (in == out) {
                continue;
            }
            const auto scale = is_scales_mode ? scales[i] : static_cast<float>(out) / static_cast<float>(in);

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

    register_matcher(std::make_shared<Matcher>(interpolate_pattern, matcher_name), callback);
}

}  // namespace pass
}  // namespace ov
