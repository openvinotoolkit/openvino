//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <ngraph/opsets/opset6.hpp>
#include "interp.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

OutputVector nearest_interp_v2 (const NodeContext& node) {
    auto x = node.get_ng_input("X");

    using InterpolateMode = ngraph::opset6::Interpolate::InterpolateMode;
    using CoordinateTransformMode = ngraph::opset6::Interpolate::CoordinateTransformMode;
    using Nearest_mode = ngraph::opset6::Interpolate::NearestMode;
    using InterpolateAttrs = ngraph::opset6::Interpolate::InterpolateAttrs;
    using ShapeCalcMode = ngraph::opset6::Interpolate::ShapeCalcMode;

    InterpolateAttrs attrs;

    attrs.mode = InterpolateMode::nearest; //HARDCODE

    auto out_w = node.get_attribute<int>("out_w");
    auto out_h = node.get_attribute<int>("out_h");
    auto scale = node.get_attribute<std::vector<float>>("scale");
    if (out_w <= 0 || out_h <= 0) {
        attrs.shape_calculation_mode = ShapeCalcMode::scales;
    }
    else {
        attrs.shape_calculation_mode = ShapeCalcMode::sizes;
    }

    auto target_spatial_shape =
        ngraph::opset6::Constant::create<int64_t>(element::i64, Shape{2}, {out_h, out_w});
    auto scales = ngraph::opset6::Constant::create<float>(element::f32, Shape{2}, std::vector<float>(scale.begin(), scale.end()));  
    auto axes = ngraph::opset6::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});  
    
    attrs.coordinate_transformation_mode = CoordinateTransformMode::asymmetric; //HARDCODE
    attrs.nearest_mode = Nearest_mode::round_prefer_floor; //HARDCODE
    attrs.antialias = false;  //HARDCODE
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    return {std::make_shared<ngraph::opset6::Interpolate>(x, target_spatial_shape, scales, axes, attrs)};
}

}}}}