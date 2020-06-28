// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeScaleShift(const ngraph::Output<Node> &in,
                                             const element::Type &type,
                                             const std::vector<float> &scales,
                                             const std::vector<float> &shifts) {
    bool randomScales = scales.empty();
    bool randomShifts = shifts.empty();

    auto shape = in.get_shape();
    Shape scalesShape(shape.size(), 1);
    scalesShape[1] = shape[1];

    auto scalesNode = makeConstant(type, scalesShape, scales, randomScales);
    auto shiftsNode = makeConstant(type, scalesShape, shifts, randomShifts);

    auto multiply = std::make_shared<opset1::Multiply>(in, scalesNode);
    auto add = std::make_shared<opset1::Add>(multiply, shiftsNode);

    return add;
}

}  // namespace builder
}  // namespace ngraph
