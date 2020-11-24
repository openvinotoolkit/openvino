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
    auto interpolate1 = ngraph::pattern::wrap_type<ngraph::opset1::Interpolate>({ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto interpolationV0 = std::dynamic_pointer_cast<ngraph::op::v0::Interpolate>(m.get_match_root());
        if (!interpolationV0) {
            return false;
        }

        auto& inp_shape = interpolationV0->get_input_shape(0);
        auto& out_shape = interpolationV0->get_output_shape(0);
        auto attrsV0 = interpolationV0->get_attrs();

        std::vector<float> scales;
        for (std::size_t i = 0; i < inp_shape.size(); ++i) {
            scales.emplace_back(static_cast<float>(inp_shape.at(i))/out_shape.at(i));
        }
        auto scalesConstant = ngraph::op::Constant::create(ngraph::element::f32, {scales.size()}, scales);
        auto axisConstant = ngraph::op::Constant::create(ngraph::element::i64, {attrsV0.axes.size()},
                                                         std::vector<std::size_t>{attrsV0.axes.begin(), attrsV0.axes.end()});

        ngraph::op::v4::Interpolate::InterpolateAttrs attrsV4;

        if (attrsV0.mode == "nearest") {
            attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::nearest;
        } else if (attrsV0.mode == "linear") {
            attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear;
        } else if (attrsV0.mode == "cubic") {
            attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::cubic;
        } else if (attrsV0.mode == "linear_onnx") {
            attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx;
        } else {
            return false;
        }
        attrsV4.shape_calculation_mode = ngraph::op::v4::Interpolate::ShapeCalcMode::sizes;
        attrsV4.nearest_mode = ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor;
        attrsV4.pads_begin = attrsV0.pads_begin;
        attrsV4.pads_end = attrsV0.pads_end;
        attrsV4.antialias = attrsV0.antialias;
        attrsV4.coordinate_transformation_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel;
        attrsV4.cube_coeff = -0.75f;
        if (attrsV0.align_corners) {
            attrsV4.coordinate_transformation_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners;
        }

        auto interpolateV4 = std::make_shared<ngraph::op::v4::Interpolate>(interpolationV0->input_value(0), interpolationV0->input_value(1),
                                                                           scalesConstant, axisConstant, attrsV4);

        interpolateV4->set_friendly_name(interpolationV0->get_friendly_name() + ".V4");
        ngraph::copy_runtime_info(interpolationV0, interpolateV4);
        ngraph::replace_node(interpolationV0, interpolateV4);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate1, "ConvertInterpolate1ToInterpolate4");
    this->register_matcher(m, callback);
}
