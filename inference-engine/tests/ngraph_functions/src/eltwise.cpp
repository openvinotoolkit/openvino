// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <memory>
#include <ngraph/opsets/opset3.hpp>
#include "ngraph_functions/utils/ngraph_helpers.hpp"


namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeEltwise(const ngraph::Output<Node> &in0,
                                          const ngraph::Output<Node> &in1,
                                          ngraph::helpers::EltwiseTypes eltwiseType) {
    switch (eltwiseType) {
        case ngraph::helpers::EltwiseTypes::ADD:
            return std::make_shared<ngraph::opset3::Add>(in0, in1);
        case ngraph::helpers::EltwiseTypes::SUBTRACT:
            return std::make_shared<ngraph::opset3::Subtract>(in0, in1);
        case ngraph::helpers::EltwiseTypes::MULTIPLY:
            return std::make_shared<ngraph::opset3::Multiply>(in0, in1);
        case ngraph::helpers::EltwiseTypes::DIVIDE:
            return std::make_shared<ngraph::opset3::Divide>(in0, in1);
        case ngraph::helpers::EltwiseTypes::SQUARED_DIFF:
            return std::make_shared<ngraph::opset3::SquaredDifference>(in0, in1);
        case ngraph::helpers::EltwiseTypes::POWER:
            return std::make_shared<ngraph::opset3::Power>(in0, in1);
        case ngraph::helpers::EltwiseTypes::FLOOR_MOD:
            return std::make_shared<ngraph::opset3::FloorMod>(in0, in1);
        default: {
            throw std::runtime_error("Incorrect type of Eltwise operation");
        }
    }
}

}  // namespace builder
}  // namespace ngraph