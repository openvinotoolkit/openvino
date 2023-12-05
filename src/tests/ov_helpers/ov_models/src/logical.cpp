// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "common_test_utils/test_enums.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeLogical(const ov::Output<Node>& in0,
                                      const ov::Output<Node>& in1,
                                      ov::test::utils::LogicalTypes logicalType) {
    switch (logicalType) {
    case ov::test::utils::LogicalTypes::LOGICAL_AND:
        return std::make_shared<ov::op::v1::LogicalAnd>(in0, in1);
    case ov::test::utils::LogicalTypes::LOGICAL_OR:
        return std::make_shared<ov::op::v1::LogicalOr>(in0, in1);
    case ov::test::utils::LogicalTypes::LOGICAL_NOT:
        return std::make_shared<ov::op::v1::LogicalNot>(in0);
    case ov::test::utils::LogicalTypes::LOGICAL_XOR:
        return std::make_shared<ov::op::v1::LogicalXor>(in0, in1);
    default: {
        throw std::runtime_error("Incorrect type of Logical operation");
    }
    }
}

std::shared_ptr<ov::Node> makeLogical(const ov::ParameterVector& inputs, ov::test::utils::LogicalTypes logicalType) {
    switch (logicalType) {
    case ov::test::utils::LogicalTypes::LOGICAL_AND:
        return std::make_shared<ov::op::v1::LogicalAnd>(inputs[0], inputs[1]);
    case ov::test::utils::LogicalTypes::LOGICAL_OR:
        return std::make_shared<ov::op::v1::LogicalOr>(inputs[0], inputs[1]);
    case ov::test::utils::LogicalTypes::LOGICAL_NOT:
        return std::make_shared<ov::op::v1::LogicalNot>(inputs[0]);
    case ov::test::utils::LogicalTypes::LOGICAL_XOR:
        return std::make_shared<ov::op::v1::LogicalXor>(inputs[0], inputs[1]);
    default: {
        throw std::runtime_error("Incorrect type of Logical operation");
    }
    }
}

}  // namespace builder
}  // namespace ngraph
