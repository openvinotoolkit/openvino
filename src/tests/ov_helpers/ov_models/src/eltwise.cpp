// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/opsets/opset13.hpp>

#include "common_test_utils/test_enums.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeEltwise(const ov::Output<Node>& in0,
                                      const ov::Output<Node>& in1,
                                      ov::test::utils::EltwiseTypes eltwiseType) {
    switch (eltwiseType) {
    case ov::test::utils::EltwiseTypes::ADD:
        return std::make_shared<ov::op::v1::Add>(in0, in1);
    case ov::test::utils::EltwiseTypes::SUBTRACT:
        return std::make_shared<ov::op::v1::Subtract>(in0, in1);
    case ov::test::utils::EltwiseTypes::MULTIPLY:
        return std::make_shared<ov::op::v1::Multiply>(in0, in1);
    case ov::test::utils::EltwiseTypes::DIVIDE:
        return std::make_shared<ov::op::v1::Divide>(in0, in1);
    case ov::test::utils::EltwiseTypes::SQUARED_DIFF:
        return std::make_shared<ov::op::v0::SquaredDifference>(in0, in1);
    case ov::test::utils::EltwiseTypes::POWER:
        return std::make_shared<ov::op::v1::Power>(in0, in1);
    case ov::test::utils::EltwiseTypes::FLOOR_MOD:
        return std::make_shared<ov::op::v1::FloorMod>(in0, in1);
    case ov::test::utils::EltwiseTypes::MOD:
        return std::make_shared<ov::op::v1::Mod>(in0, in1);
    case ov::test::utils::EltwiseTypes::ERF:
        return std::make_shared<ov::op::v0::Erf>(in0);
    case ngraph::helpers::EltwiseTypes::BITWISE_AND:
        return std::make_shared<ov::op::v13::BitwiseAnd>(in0, in1);
    case ngraph::helpers::EltwiseTypes::BITWISE_NOT:
        return std::make_shared<ov::op::v13::BitwiseNot>(in0);
    case ngraph::helpers::EltwiseTypes::BITWISE_OR:
        return std::make_shared<ov::op::v13::BitwiseOr>(in0, in1);
    case ngraph::helpers::EltwiseTypes::BITWISE_XOR:
        return std::make_shared<ov::op::v13::BitwiseXor>(in0, in1);
    default: {
        throw std::runtime_error("Incorrect type of Eltwise operation");
    }
    }
}

}  // namespace builder
}  // namespace ngraph
