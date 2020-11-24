//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "transformations/convert_interpolate_v0_to_v4.hpp"
#include "ngraph/op/interpolate.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/rt_info.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertInterpolateV1toV4, "ConvertInterpolateV1toV4", 0);

pass::ConvertInterpolateV1toV4::ConvertInterpolateV1toV4() {
//     register_matcher(
//         std::make_shared<ngraph::pattern::Matcher>(
//             std::make_shared<ngraph::op::v0::Interpolate>(
//                 ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::op::v0::InterpolateAttrs{}),
//             "ConvertInterpolateV0toV4"),
//         [](ngraph::pattern::Matcher& m) {
//             auto interpolationV0 = std::dynamic_pointer_cast<ngraph::op::v0::Interpolate>(m.get_match_root());
//             if (!interpolationV0) {
//                 return false;
//             }
//
//             auto& inp_shape = interpolationV0->get_input_shape(0);
//             auto& out_shape = interpolationV0->get_output_shape(0);
//             auto attrsV0 = interpolationV0->get_attrs();
//
//             std::vector<float> scales;
//             for (std::size_t i = 0; i < inp_shape.size(); ++i) {
//                 scales.emplace_back(static_cast<float>(inp_shape.at(i))/out_shape.at(i));
//             }
//             auto scalesConstant = ngraph::op::Constant::create(
//                     ngraph::element::f32, {scales.size()}, scales);
//             auto axisConstant = ngraph::op::Constant::create(
//                     ngraph::element::i64, {attrsV0.axes.size()}, std::vector<std::size_t>{attrsV0.axes.begin(), attrsV0.axes.end()});
//
//             ngraph::op::v4::Interpolate::InterpolateAttrs attrsV4;
//
//             if (attrsV0.mode == "nearest") {
//                 attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::nearest;
//             } else if (attrsV0.mode == "linear") {
//                 attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear;
//             } else if (attrsV0.mode == "cubic") {
//                 attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::cubic;
//             } else if (attrsV0.mode == "linear_onnx") {
//                 attrsV4.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx;
//             } else {
//                 return false;
//             }
//             attrsV4.shape_calculation_mode = ngraph::op::v4::Interpolate::ShapeCalcMode::sizes;
//             attrsV4.pads_begin = attrsV0.pads_begin;
//             attrsV4.pads_end = attrsV0.pads_end;
//             if (attrsV0.align_corners) {
//                 attrsV4.coordinate_transformation_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners;
//             }
//
//             auto interpolateV4 = std::make_shared<ngraph::op::v4::Interpolate>(
//                 interpolationV0->input_value(0),
//                 interpolationV0->input_value(1),
//                 scalesConstant,
//                 axisConstant,
//                 attrsV4);
//
//             interpolateV4->set_friendly_name(interpolationV0->get_friendly_name() + ".V4");
//             ngraph::copy_runtime_info(interpolationV0, interpolateV4);
//             ngraph::replace_node(interpolationV0, interpolateV4);
//             return true;
//         });
}

