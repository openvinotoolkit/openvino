// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertInterpolate1ToInterpolate4, "ConvertInterpolate1ToInterpolate4", 0);

ngraph::pass::ConvertInterpolate1ToInterpolate4::ConvertInterpolate1ToInterpolate4() {
    MATCHER_SCOPE(ConvertInterpolate1ToInterpolate4);
    auto interpolate1 = ngraph::pattern::wrap_type<ngraph::opset1::Interpolate>({pattern::any_input(pattern::has_static_shape()), pattern::any_input()});
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto interpolationV0 = std::dynamic_pointer_cast<ngraph::opset1::Interpolate>(m.get_match_root());
        if (!interpolationV0) {
            return false;
        }

        auto& inp_partial_shape = interpolationV0->get_input_partial_shape(0);
        auto& out_shape = interpolationV0->get_output_shape(0);
        auto attrsV0 = interpolationV0->get_attrs();

        std::vector<float> scales(attrsV0.axes.size(), 1.0f);
        auto inp_shape = inp_partial_shape.to_shape();
        size_t i = 0;
        for (std::size_t axis : attrsV0.axes) {
            scales[i] = static_cast<float>(out_shape.at(axis))/inp_shape.at(axis);
            i++;
        }

        auto input_shape_rank = inp_partial_shape.rank().get_length();
        auto scalesConstant = ngraph::op::Constant::create(ngraph::element::f32, {scales.size()}, scales);
        auto axisConstant = ngraph::op::Constant::create(ngraph::element::i64, {attrsV0.axes.size()},
                                                         std::vector<std::size_t>{attrsV0.axes.begin(), attrsV0.axes.end()});

        ngraph::opset4::Interpolate::InterpolateAttrs attrsV4;

        if (attrsV0.mode == "nearest") {
            attrsV4.mode = ngraph::opset4::Interpolate::InterpolateMode::nearest;
        } else if (attrsV0.mode == "linear") {
            // If we write only
            //    attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear;
            // instead of a conditional statements below when attrsV0.mode == "linear",
            // then we have a performance drop, because CPU have no optimized
            // version of the 'linear' mode.
            // TODO: delete this conditional statement, when CPU will have
            // optimized version of the 'linear' mode.
            if (input_shape_rank < 5) {
                attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx;
            } else if (input_shape_rank == 5) {
                attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear;
            } else {
                return false;
            }
        } else if (attrsV0.mode == "cubic") {
            attrsV4.mode = ngraph::opset4::Interpolate::InterpolateMode::cubic;
        } else if (attrsV0.mode == "linear_onnx") {
            attrsV4.mode = ngraph::opset4::Interpolate::InterpolateMode::linear_onnx;
        } else {
            return false;
        }
        attrsV4.shape_calculation_mode = ngraph::opset4::Interpolate::ShapeCalcMode::sizes;
        attrsV4.nearest_mode = ngraph::opset4::Interpolate::NearestMode::simple;
        attrsV4.pads_begin = attrsV0.pads_begin;
        attrsV4.pads_end = attrsV0.pads_end;
        attrsV4.antialias = attrsV0.antialias;
        attrsV4.coordinate_transformation_mode = ngraph::opset4::Interpolate::CoordinateTransformMode::asymmetric;
        attrsV4.cube_coeff = -0.75f;
        if (attrsV0.align_corners) {
            attrsV4.coordinate_transformation_mode = ngraph::opset4::Interpolate::CoordinateTransformMode::align_corners;
        } else if ((attrsV4.mode == ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx ||
                    attrsV4.mode == ngraph::op::v4::Interpolate::InterpolateMode::linear) &&
                    std::all_of(attrsV4.pads_begin.begin(), attrsV4.pads_begin.end(), [](size_t i){return i == 0;}) &&
                    std::all_of(attrsV4.pads_end.begin(), attrsV4.pads_end.end(), [](size_t i){return i == 0;}) &&
                    !(input_shape_rank - 2 == 2 && attrsV0.axes == AxisSet{2, 3})) {
            attrsV4.coordinate_transformation_mode = ngraph::opset4::Interpolate::CoordinateTransformMode::half_pixel;
        }

        auto interpolateV4 = std::make_shared<ngraph::opset4::Interpolate>(interpolationV0->input_value(0), interpolationV0->input_value(1),
                                                                           scalesConstant, axisConstant, attrsV4);

        interpolateV4->set_friendly_name(interpolationV0->get_friendly_name());
        ngraph::copy_runtime_info(interpolationV0, interpolateV4);
        ngraph::replace_node(interpolationV0, interpolateV4);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate1, matcher_name);
    this->register_matcher(m, callback);
}
