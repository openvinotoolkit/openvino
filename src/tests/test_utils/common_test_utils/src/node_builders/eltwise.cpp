// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/eltwise.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_left_shift.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_eltwise(const ov::Output<Node>& in0,
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
    case ov::test::utils::EltwiseTypes::BITWISE_AND:
        return std::make_shared<ov::op::v13::BitwiseAnd>(in0, in1);
    case ov::test::utils::EltwiseTypes::BITWISE_NOT:
        return std::make_shared<ov::op::v13::BitwiseNot>(in0);
    case ov::test::utils::EltwiseTypes::BITWISE_OR:
        return std::make_shared<ov::op::v13::BitwiseOr>(in0, in1);
    case ov::test::utils::EltwiseTypes::BITWISE_XOR:
        return std::make_shared<ov::op::v13::BitwiseXor>(in0, in1);
    case ov::test::utils::EltwiseTypes::RIGHT_SHIFT:
        return std::make_shared<ov::op::v15::BitwiseRightShift>(in0, in1);
    case ov::test::utils::EltwiseTypes::LEFT_SHIFT:
        return std::make_shared<ov::op::v15::BitwiseLeftShift>(in0, in1);
    default: {
        OPENVINO_THROW("Incorrect type of Eltwise operation");
    }
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
