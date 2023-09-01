// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeMinMax(const ov::Output<Node> &in1,
                                         const ov::Output<Node> &in2,
                                         ov::helpers::MinMaxOpType opType) {
    switch (opType) {
        case ov::helpers::MinMaxOpType::MINIMUM:
            return std::make_shared<ov::opset3::Minimum>(in1, in2);
        case ov::helpers::MinMaxOpType::MAXIMUM:
            return std::make_shared<ov::opset3::Maximum>(in1, in2);
        default:
            throw std::logic_error("Unsupported operation type");
    }
}

}  // namespace builder
}  // namespace ov
