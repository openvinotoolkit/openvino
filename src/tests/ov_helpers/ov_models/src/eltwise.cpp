// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/opsets/opset3.hpp>
#include "ov_models/utils/ov_helpers.hpp"


namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeEltwise(const ov::Output<Node> &in0,
                                          const ov::Output<Node> &in1,
                                          ov::helpers::EltwiseTypes eltwiseType) {
    switch (eltwiseType) {
        case ov::helpers::EltwiseTypes::ADD:
            return std::make_shared<ov::opset3::Add>(in0, in1);
        case ov::helpers::EltwiseTypes::SUBTRACT:
            return std::make_shared<ov::opset3::Subtract>(in0, in1);
        case ov::helpers::EltwiseTypes::MULTIPLY:
            return std::make_shared<ov::opset3::Multiply>(in0, in1);
        case ov::helpers::EltwiseTypes::DIVIDE:
            return std::make_shared<ov::opset3::Divide>(in0, in1);
        case ov::helpers::EltwiseTypes::SQUARED_DIFF:
            return std::make_shared<ov::opset3::SquaredDifference>(in0, in1);
        case ov::helpers::EltwiseTypes::POWER:
            return std::make_shared<ov::opset3::Power>(in0, in1);
        case ov::helpers::EltwiseTypes::FLOOR_MOD:
            return std::make_shared<ov::opset3::FloorMod>(in0, in1);
        case ov::helpers::EltwiseTypes::MOD:
            return std::make_shared<ov::opset3::Mod>(in0, in1);
        case ov::helpers::EltwiseTypes::ERF:
            return std::make_shared<ov::opset1::Erf>(in0);
        default: {
            throw std::runtime_error("Incorrect type of Eltwise operation");
        }
    }
}

}  // namespace builder
}  // namespace ov
