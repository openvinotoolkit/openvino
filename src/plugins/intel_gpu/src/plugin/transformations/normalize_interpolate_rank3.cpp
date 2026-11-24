// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_interpolate_rank3.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {
namespace {
using Interpolate = ov::op::util::InterpolateBase;
using Constant = ov::op::v0::Constant;

size_t axes_port(const std::shared_ptr<Interpolate>& op) {
    return ov::is_type<ov::op::v4::Interpolate>(op) ? 3 : 2;
}

size_t values_port(const std::shared_ptr<Interpolate>& op) {
    return ov::is_type<ov::op::v4::Interpolate>(op) && op->get_attrs().shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES ? 2 : 1;
}

std::vector<int64_t> interpolation_axes(const std::shared_ptr<Interpolate>& op) {
    if (op->get_input_size() <= axes_port(op))
        return {0, 1, 2};
    const auto constant = ov::as_type_ptr<Constant>(op->get_input_node_shared_ptr(axes_port(op)));
    if (!constant)
        return {};
    auto axes = constant->cast_vector<int64_t>();
    for (auto& axis : axes) {
        if (axis < 0)
            axis += 3;
    }
    return axes;
}

bool can_normalize(const ov::Output<ov::Node>& output) {
    const auto op = ov::as_type_ptr<Interpolate>(output.get_node_shared_ptr());
    OPENVINO_ASSERT(op, "NormalizeInterpolateRank3 expects Interpolate-4 or Interpolate-11");
    const auto& shape = op->get_input_partial_shape(0);
    const auto& attrs = op->get_attrs();
    if (shape.rank() != 3 ||
        std::any_of(attrs.pads_begin.begin(),
                    attrs.pads_begin.end(),
                    [](size_t pad) {
                        return pad != 0;
                    }) ||
        std::any_of(attrs.pads_end.begin(), attrs.pads_end.end(), [](size_t pad) {
            return pad != 0;
        })) {
        return false;
    }

    const auto axes = interpolation_axes(op);
    const auto values = ov::as_type_ptr<Constant>(op->get_input_node_shared_ptr(values_port(op)));
    if (!values || axes.empty() || values->get_shape() != ov::Shape{axes.size()} || std::count(axes.begin(), axes.end(), 2) != 1) {
        return false;
    }

    std::vector<bool> seen(3, false);
    const bool scales_mode = attrs.shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES;
    const auto scales = scales_mode ? values->cast_vector<double>() : std::vector<double>{};
    const auto sizes = scales_mode ? std::vector<int64_t>{} : values->cast_vector<int64_t>();
    for (size_t i = 0; i < axes.size(); ++i) {
        const auto axis = axes[i];
        if (axis < 0 || axis >= 3 || seen[axis])
            return false;
        seen[axis] = true;
        if (scales_mode) {
            if (!std::isfinite(scales[i]) || scales[i] <= 0 || (axis != 2 && scales[i] != 1.0))
                return false;
        } else {
            if (sizes[i] <= 0 || (axis != 2 && (shape[axis].is_dynamic() || shape[axis].get_length() != sizes[i]))) {
                return false;
            }
        }
    }
    return true;
}
}  // namespace

NormalizeInterpolateRank3::NormalizeInterpolateRank3() {
    MATCHER_SCOPE(NormalizeInterpolateRank3);
    const auto interpolate =
        ov::pass::pattern::wrap_type<ov::op::v4::Interpolate, ov::op::v11::Interpolate>(can_normalize,
                                                                                        {{"mode", std::string("linear_onnx")}, {"antialias", false}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto op = ov::as_type_ptr<Interpolate>(pattern_map.at(interpolate).get_node_shared_ptr());
        OPENVINO_ASSERT(op, "NormalizeInterpolateRank3 expects Interpolate-4 or Interpolate-11");
        if (transformation_callback(op))
            return false;

        const auto axes = interpolation_axes(op);
        const auto last_index = std::distance(axes.begin(), std::find(axes.begin(), axes.end(), 2));
        const auto values = ov::as_type_ptr<Constant>(op->get_input_node_shared_ptr(values_port(op)));
        OPENVINO_ASSERT(values, "NormalizeInterpolateRank3 expects constant active scales/sizes");
        const bool scales_mode = op->get_attrs().shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES;
        // Double preserves every supported scale type; keep sizes in the integer domain.
        const auto spatial_values =
            scales_mode ? Constant::create(values->get_element_type(), {2}, std::vector<double>{1, values->cast_vector<double>()[last_index]})
                        : Constant::create(values->get_element_type(), {2}, std::vector<uint64_t>{1, values->cast_vector<uint64_t>()[last_index]});

        const auto singleton_axis = Constant::create(ov::element::i64, {1}, {2});
        const auto spatial_axes = Constant::create(ov::element::i64, {2}, {2, 3});
        const auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(op->input_value(0), singleton_axis);
        auto attrs = op->get_attrs();
        attrs.pads_begin = attrs.pads_end = std::vector<size_t>(4, 0);
        std::shared_ptr<ov::Node> normalized;
        ov::NodeVector new_nodes{spatial_values, singleton_axis, spatial_axes, unsqueeze};
        if (ov::is_type<ov::op::v4::Interpolate>(op)) {
            const auto unused = Constant::create(op->get_input_element_type(scales_mode ? 1 : 2), {2}, {1, 1});
            new_nodes.push_back(unused);
            normalized = std::make_shared<ov::op::v4::Interpolate>(unsqueeze,
                                                                   scales_mode ? unused : spatial_values,
                                                                   scales_mode ? spatial_values : unused,
                                                                   spatial_axes,
                                                                   attrs);
        } else {
            normalized = std::make_shared<ov::op::v11::Interpolate>(unsqueeze, spatial_values, spatial_axes, attrs);
        }

        const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(normalized, singleton_axis);
        squeeze->set_friendly_name(op->get_friendly_name());
        new_nodes.insert(new_nodes.end(), {normalized, squeeze});
        ov::copy_runtime_info(op, new_nodes);
        ov::replace_node(op, squeeze);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(interpolate, matcher_name), callback);
}
}  // namespace ov::intel_gpu
