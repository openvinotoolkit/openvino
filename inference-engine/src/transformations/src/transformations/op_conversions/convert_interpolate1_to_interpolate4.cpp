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
    auto interpolate1 = ngraph::pattern::wrap_type<ngraph::opset1::Interpolate>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto interpolate1 = std::dynamic_pointer_cast<ngraph::opset1::Interpolate>(m.get_match_root());
        if (!interpolate1)
            return false;

        auto data_node = interpolate1->input_value(0);
        auto out_shape_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(interpolate1->input_value(1).get_node_shared_ptr());
        auto interpolate_attrs = interpolate1->get_attrs();
        auto input_shape = data_node.get_shape();

        // attrs
        auto mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode();
        if (interpolate_attrs.mode == "nearest") {
            mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode::nearest;
        } else if (interpolate_attrs.mode == "cubic") {
            mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode::cubic;
        } else if (interpolate_attrs.mode == "linear") {
            if (input_shape.size() < 5) {
                mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx;
            } else if (input_shape.size() == 5) {
                mode_v4 = ngraph::op::v4::Interpolate::InterpolateMode::linear;
            } else {
                return false;
            }
        } else {
            return false;
        }
        auto nearest_mode_v4 = ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor;
        auto shape_calculation_mode_v4 = ngraph::op::v4::Interpolate::ShapeCalcMode::sizes;
        auto coordinate_transformation_mode_v4 = interpolate_attrs.align_corners ? ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners :
                                                ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel;
        auto interpolate4_attr = ngraph::op::v4::Interpolate::InterpolateAttrs(mode_v4, shape_calculation_mode_v4,
            interpolate_attrs.pads_begin, interpolate_attrs.pads_end,
            coordinate_transformation_mode_v4, nearest_mode_v4, interpolate_attrs.antialias, -0.75);

        // input
        auto scales_node = ngraph::opset4::Constant::create(ngraph::element::f32, Shape{}, {0});
        auto axes = interpolate_attrs.axes.to_vector();
        auto axes_const = ngraph::opset4::Constant(ngraph::element::Type_t::i64, {axes.size()}, axes);
        auto axes_node = std::make_shared<ngraph::opset4::Constant>(axes_const);

        auto interpolate4 = std::make_shared<ngraph::opset4::Interpolate>(data_node, out_shape_node, scales_node, axes_node, interpolate4_attr);

        interpolate4->set_friendly_name(interpolate1->get_friendly_name());
        ngraph::copy_runtime_info(interpolate1, interpolate4);
        ngraph::replace_node(interpolate1, interpolate4);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate1, "ConvertInterpolate1ToInterpolate4");
    this->register_matcher(m, callback);
}