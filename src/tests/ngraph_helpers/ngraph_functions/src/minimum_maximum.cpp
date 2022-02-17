// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeMinMax(const ngraph::Output<Node> &in1,
                                         const ngraph::Output<Node> &in2,
                                         ngraph::helpers::MinMaxOpType opType) {
    switch (opType) {
        case ngraph::helpers::MinMaxOpType::MINIMUM:
            return std::make_shared<ngraph::opset3::Minimum>(in1, in2);
        case ngraph::helpers::MinMaxOpType::MAXIMUM:
            return std::make_shared<ngraph::opset3::Maximum>(in1, in2);
        default:
            throw std::logic_error("Unsupported operation type");
    }
}

}  // namespace builder
}  // namespace ngraph
