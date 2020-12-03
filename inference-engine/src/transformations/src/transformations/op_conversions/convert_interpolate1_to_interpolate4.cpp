// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertInterpolate1ToInterpolate4, "ConvertInterpolate1ToInterpolate4", 0);

ngraph::pass::ConvertInterpolate1ToInterpolate4::ConvertInterpolate1ToInterpolate4() {
    auto interpolate1 = ngraph::pattern::wrap_type<ngraph::opset1::Interpolate>({pattern::any_input(pattern::has_static_rank()), pattern::any_input()});
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto interpolate1 = std::dynamic_pointer_cast<ngraph::opset1::Interpolate>(m.get_match_root());
        if (!interpolate1)
            return false;

        auto interpolate_attrs = interpolate1->get_attrs();
        auto input_shape_rank = interpolate1->input(0).get_partial_shape().rank().get_length();

        // attrs
        auto mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode();
        if (interpolate_attrs.mode == "nearest") {
            mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode::nearest;
        } else if (interpolate_attrs.mode == "cubic") {
            mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode::cubic;
        } else if (interpolate_attrs.mode == "linear") {
            if (input_shape_rank < 5) {
                mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx;
            } else if (input_shape_rank == 5) {
                mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode::linear;
            } else {
                return false;
            }
        } else {
            return false;
        }
        auto nearest_mode_v4 = ngraph::op::v4::Interpolate::NearestMode::floor;
        auto shape_calculation_mode_v4 = ngraph::op::v4::Interpolate::ShapeCalcMode::sizes;
        auto coordinate_transformation_mode_v4 = interpolate_attrs.align_corners ? ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners :
                                                ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric;
        auto interpolate4_attr = ngraph::op::v4::Interpolate::InterpolateAttrs(mode_v4, shape_calculation_mode_v4,
            interpolate_attrs.pads_begin, interpolate_attrs.pads_end,
            coordinate_transformation_mode_v4, nearest_mode_v4, interpolate_attrs.antialias, -0.75);

        // input
        auto axes = interpolate_attrs.axes.to_vector();
        auto axes_node = ngraph::opset4::Constant::create(element::i64, {axes.size()}, axes);
        auto default_scales = std::vector<float>(axes.size(), 1.f);
        auto scales_node = ngraph::opset4::Constant::create(element::f32, {axes.size()}, default_scales);

        auto interpolate4 = std::make_shared<ngraph::opset4::Interpolate>(interpolate1->input_value(0), interpolate1->input_value(1),
            scales_node, axes_node, interpolate4_attr);

        interpolate4->set_friendly_name(interpolate1->get_friendly_name());
        ngraph::copy_runtime_info(interpolate1, interpolate4);
        ngraph::replace_node(interpolate1, interpolate4);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate1, "ConvertInterpolate1ToInterpolate4");
    this->register_matcher(m, callback);
}