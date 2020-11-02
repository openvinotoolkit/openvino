// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeInterpolate(const std::vector<ngraph::Output<Node>>& in, bool antialias) {
    printf("makeInterpolate\n");
    std::vector<size_t> padBegin(1, 0), padEnd(1, 0);
    float coeff = -0.75;
    ngraph::op::v4::Interpolate::InterpolateMode mode = ngraph::op::v4::Interpolate::InterpolateMode::nearest;
    ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalc = ngraph::op::v4::Interpolate::ShapeCalcMode::scales;
    ngraph::op::v4::Interpolate::CoordinateTransformMode coordTrans = ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel;
    ngraph::op::v4::Interpolate::NearestMode nearestMode = ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor;

    const ngraph::op::v4::Interpolate::InterpolateAttrs interpolateAttributes(mode, shapeCalc, padBegin, padEnd, coordTrans, nearestMode, antialias, coeff);

    return std::make_shared<ngraph::opset4::Interpolate>(in[0], in[1], in[2], in[3], interpolateAttributes);
}

}  // namespace builder
}  // namespace ngraph
