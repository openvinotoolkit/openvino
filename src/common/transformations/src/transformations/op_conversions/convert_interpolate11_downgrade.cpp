// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"

#include <array>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

namespace {
// v4_sizes, v4_scales
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> make_v4_inputs(
    const std::shared_ptr<ov::op::v11::Interpolate>& interpolate) {
    ov::pass::NodeRegistry registry;
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> ret;
    std::shared_ptr<ov::Node> broadcast_shape;

    if (interpolate->get_input_size() == 3) {
        // broadcast dummy constant to the shape of axes
        broadcast_shape = registry.add(ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(interpolate->input_value(2)));
    } else {
        // broadcast dummy constant to the rank of data
        broadcast_shape = registry.add(ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(interpolate->input_value(0)));
        broadcast_shape = registry.add(ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(broadcast_shape));
    }

    if (interpolate->get_attrs().shape_calculation_mode == ov::op::util::InterpolateBase::ShapeCalcMode::SCALES) {
        ret.second = interpolate->input_value(1);
        std::shared_ptr<ov::Node> sizes_input = registry.make<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 1);
        sizes_input = registry.add(ov::op::util::make_try_fold<ov::op::v3::Broadcast>(sizes_input, broadcast_shape));
        ret.first = sizes_input;
    } else {
        ret.first = interpolate->input_value(1);
        std::shared_ptr<ov::Node> scales_input =
            registry.make<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 1.0f);
        scales_input = registry.add(ov::op::util::make_try_fold<ov::op::v3::Broadcast>(scales_input, broadcast_shape));
        ret.second = scales_input;
    }

    copy_runtime_info(interpolate, registry.get());

    return ret;
}
}  // namespace

ov::pass::ConvertInterpolate11ToInterpolate4::ConvertInterpolate11ToInterpolate4() {
    MATCHER_SCOPE(ConvertInterpolate11ToInterpolate4);

    const auto interpolate_v11_pattern = pattern::wrap_type<ov::op::v11::Interpolate>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto v4_compatible_interpolation_mode = [](const op::util::InterpolateBase::InterpolateMode mode) {
            constexpr std::array<op::util::InterpolateBase::InterpolateMode, 4> allowed_modes = {
                op::util::InterpolateBase::InterpolateMode::NEAREST,
                op::util::InterpolateBase::InterpolateMode::LINEAR,
                op::util::InterpolateBase::InterpolateMode::LINEAR_ONNX,
                op::util::InterpolateBase::InterpolateMode::CUBIC};

            return std::find(std::begin(allowed_modes), std::end(allowed_modes), mode) != std::end(allowed_modes);
        };

        const auto interpolate_v11 = ov::as_type_ptr<ov::op::v11::Interpolate>(m.get_match_root());
        if (!interpolate_v11 || !v4_compatible_interpolation_mode(interpolate_v11->get_attrs().mode) ||
            transformation_callback(interpolate_v11)) {
            return false;
        }

        // downgrade only if the interpolation mode used to create v11 is supported by v4
        std::shared_ptr<ov::op::v4::Interpolate> interpolate_v4;
        ov::Output<ov::Node> v4_input_output_shape;
        ov::Output<ov::Node> v4_input_scales;
        std::tie(v4_input_output_shape, v4_input_scales) = make_v4_inputs(interpolate_v11);

        if (interpolate_v11->get_input_size() == 3) {  // with axes input
            interpolate_v4 = std::make_shared<ov::op::v4::Interpolate>(interpolate_v11->input_value(0),
                                                                       v4_input_output_shape,
                                                                       v4_input_scales,
                                                                       interpolate_v11->input_value(2),
                                                                       interpolate_v11->get_attrs());
        } else {
            interpolate_v4 = std::make_shared<ov::op::v4::Interpolate>(interpolate_v11->input_value(0),
                                                                       v4_input_output_shape,
                                                                       v4_input_scales,
                                                                       interpolate_v11->get_attrs());
        }

        interpolate_v4->set_friendly_name(interpolate_v11->get_friendly_name());
        copy_runtime_info(interpolate_v11, interpolate_v4);
        replace_node(interpolate_v11, interpolate_v4);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(interpolate_v11_pattern, matcher_name);
    register_matcher(m, callback);
}
