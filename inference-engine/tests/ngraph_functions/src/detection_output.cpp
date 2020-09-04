// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeDetectionOutput(const ngraph::OutputVector &inputs,
                                                  const ngraph::op::DetectionOutputAttrs& attrs) {
    if (inputs.size() == 3)
        return std::make_shared<ngraph::opset3::DetectionOutput>(inputs[0], inputs[1], inputs[2], attrs);
    else if (inputs.size() == 5)
        return std::make_shared<ngraph::opset3::DetectionOutput>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], attrs);
    else
        throw std::runtime_error("DetectionOutput layer supports only 3 or 5 inputs");
}

}  // namespace builder
}  // namespace ngraph
