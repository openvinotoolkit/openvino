// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeDetectionOutput(const ov::OutputVector& inputs,
                                              const ov::op::v0::DetectionOutput::Attributes& attrs) {
    if (inputs.size() == 3)
        return std::make_shared<ov::op::v0::DetectionOutput>(inputs[0], inputs[1], inputs[2], attrs);
    else if (inputs.size() == 5)
        return std::make_shared<ov::op::v0::DetectionOutput>(inputs[0],
                                                             inputs[1],
                                                             inputs[2],
                                                             inputs[3],
                                                             inputs[4],
                                                             attrs);
    else
        throw std::runtime_error("DetectionOutput layer supports only 3 or 5 inputs");
}

}  // namespace builder
}  // namespace ngraph
