// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/core/core.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "itt.hpp"

ngraph::pass::ConvertInterpolate1ToInterpolate4::ConvertInterpolate1ToInterpolate4() {
    MATCHER_SCOPE(ConvertInterpolate1ToInterpolate4);
    auto interpolate1 = ngraph::pattern::wrap_type<ngraph::opset1::Interpolate>(
        {pattern::any_input(pattern::has_static_rank()), pattern::any_input()});
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto interpolationV0 = std::dynamic_pointer_cast<ngraph::opset1::Interpolate>(m.get_match_root());
        if (!interpolationV0) {
            return false;
        }

        auto attrsV0 = interpolationV0->get_attrs();
        std::vector<size_t> axes{attrsV0.axes.begin(), attrsV0.axes.end()};
        const auto& out_dims = std::make_shared<opset1::Convert>(interpolationV0->input_value(1), element::f32);
        const auto& in_dims = std::make_shared<opset1::Convert>(
            ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_source(interpolationV0->input_value(0),
                                                                                   axes),
            element::f32);

        std::shared_ptr<Node> scales = std::make_shared<opset1::Divide>(out_dims, in_dims);
        if (const auto& constant = ov::get_constant_from_source(scales))
            scales = constant;
        auto axisConstant = ngraph::op::Constant::create(ngraph::element::i64, {axes.size()}, axes);

        ngraph::opset4::Interpolate::InterpolateAttrs attrsV4;
        auto input_shape_rank = interpolationV0->get_input_partial_shape(0).rank().get_length();
        if (attrsV0.mode == "nearest") {
            attrsV4.mode = ngraph::opset4::Interpolate::InterpolateMode::NEAREST;
        } else if (attrsV0.mode == "linear") {
            // If we write only
            //    attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear;
            // instead of a conditional statements below when attrsV0.mode == "linear",
            // then we have a performance drop, because CPU have no optimized
            // version of the 'linear' mode.
            // TODO: delete this conditional statement, when CPU will have
            // optimized version of the 'linear' mode.
            if (input_shape_rank < 5) {
                attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX;
            } else if (input_shape_rank == 5) {
                attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::LINEAR;
            } else {
                return false;
            }
        } else if (attrsV0.mode == "cubic") {
            attrsV4.mode = ngraph::opset4::Interpolate::InterpolateMode::CUBIC;
        } else if (attrsV0.mode == "linear_onnx") {
            attrsV4.mode = ngraph::opset4::Interpolate::InterpolateMode::LINEAR_ONNX;
        } else {
            return false;
        }
        attrsV4.shape_calculation_mode = ngraph::opset4::Interpolate::ShapeCalcMode::SIZES;
        attrsV4.nearest_mode = ngraph::opset4::Interpolate::NearestMode::SIMPLE;
        attrsV4.pads_begin = attrsV0.pads_begin;
        attrsV4.pads_end = attrsV0.pads_end;
        attrsV4.antialias = attrsV0.antialias;
        attrsV4.coordinate_transformation_mode = ngraph::opset4::Interpolate::CoordinateTransformMode::ASYMMETRIC;
        attrsV4.cube_coeff = -0.75f;
        if (attrsV0.align_corners) {
            attrsV4.coordinate_transformation_mode =
                ngraph::opset4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        } else if ((attrsV4.mode == ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX ||
                    attrsV4.mode == ngraph::op::v4::Interpolate::InterpolateMode::LINEAR) &&
                   std::all_of(attrsV4.pads_begin.begin(),
                               attrsV4.pads_begin.end(),
                               [](size_t i) {
                                   return i == 0;
                               }) &&
                   std::all_of(attrsV4.pads_end.begin(),
                               attrsV4.pads_end.end(),
                               [](size_t i) {
                                   return i == 0;
                               }) &&
                   !(input_shape_rank - 2 == 2 && attrsV0.axes == AxisSet{2, 3})) {
            attrsV4.coordinate_transformation_mode = ngraph::opset4::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        }

        auto interpolateV4 = std::make_shared<ngraph::opset4::Interpolate>(interpolationV0->input_value(0),
                                                                           interpolationV0->input_value(1),
                                                                           scales,
                                                                           axisConstant,
                                                                           attrsV4);

        interpolateV4->set_friendly_name(interpolationV0->get_friendly_name());
        ngraph::copy_runtime_info(interpolationV0, interpolateV4);
        ngraph::replace_node(interpolationV0, interpolateV4);
        MATCHER_SCOPE_ENABLE(ConvertInterpolate1ToInterpolate4);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate1, matcher_name);
    this->register_matcher(m, callback);
}
