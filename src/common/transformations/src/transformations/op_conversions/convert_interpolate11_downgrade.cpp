// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"

#include <array>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset11.hpp>
#include <openvino/opsets/opset4.hpp>

#include "itt.hpp"

ov::pass::ConvertInterpolate11ToInterpolate4::ConvertInterpolate11ToInterpolate4() {
    MATCHER_SCOPE(ConvertInterpolate11ToInterpolate4);

    const auto interpolate_v11_pattern = pattern::wrap_type<opset11::Interpolate>();

    const matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto v4_compatible_interpolation_mode = [](const op::util::InterpolateBase::InterpolateMode mode) {
            constexpr std::array<op::util::InterpolateBase::InterpolateMode, 4> allowed_modes = {
                op::util::InterpolateBase::InterpolateMode::NEAREST,
                op::util::InterpolateBase::InterpolateMode::LINEAR,
                op::util::InterpolateBase::InterpolateMode::LINEAR_ONNX,
                op::util::InterpolateBase::InterpolateMode::CUBIC};

            return std::find(std::begin(allowed_modes), std::end(allowed_modes), mode) != std::end(allowed_modes);
        };

        const auto interpolate_v11 = std::dynamic_pointer_cast<opset11::Interpolate>(m.get_match_root());
        if (!interpolate_v11 || !v4_compatible_interpolation_mode(interpolate_v11->get_attrs().mode) ||
            transformation_callback(interpolate_v11)) {
            return false;
        }

        // downgrade only if the interpolation mode used to create v11 is supported by v4
        std::shared_ptr<ov::opset4::Interpolate> interpolate_v4;
        ov::Output<ov::Node> v4_input_output_shape;
        ov::Output<ov::Node> v4_input_scales;

        if (interpolate_v11->get_attrs().shape_calculation_mode ==
            ov::op::util::InterpolateBase::ShapeCalcMode::SCALES) {
            v4_input_scales = interpolate_v11->input_value(1);
            v4_input_output_shape = opset4::Constant::create(element::i32, Shape{1}, {1});
            copy_runtime_info(interpolate_v11, v4_input_output_shape.get_node_shared_ptr());
        } else {
            v4_input_output_shape = interpolate_v11->input_value(1);
            v4_input_scales = opset4::Constant::create(element::f32, Shape{1}, {1.0f});
            copy_runtime_info(interpolate_v11, v4_input_scales.get_node_shared_ptr());
        }

        if (interpolate_v11->get_input_size() == 3) {  // with axes input
            interpolate_v4 = std::make_shared<ov::opset4::Interpolate>(interpolate_v11->input_value(0),
                                                                       v4_input_output_shape,
                                                                       v4_input_scales,
                                                                       interpolate_v11->input_value(2),
                                                                       interpolate_v11->get_attrs());
        } else {
            interpolate_v4 = std::make_shared<ov::opset4::Interpolate>(interpolate_v11->input_value(0),
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
