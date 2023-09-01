// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/opsets/opset3.hpp>
#include "ov_models/utils/ov_helpers.hpp"

namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeLogical(const ov::Output<Node> &in0,
                                          const ov::Output<Node> &in1,
                                          ov::helpers::LogicalTypes logicalType) {
    switch (logicalType) {
        case ov::helpers::LogicalTypes::LOGICAL_AND:
            return std::make_shared<ov::opset3::LogicalAnd>(in0, in1);
        case ov::helpers::LogicalTypes::LOGICAL_OR:
            return std::make_shared<ov::opset3::LogicalOr>(in0, in1);
        case ov::helpers::LogicalTypes::LOGICAL_NOT:
            return std::make_shared<ov::opset3::LogicalNot>(in0);
        case ov::helpers::LogicalTypes::LOGICAL_XOR:
            return std::make_shared<ov::opset3::LogicalXor>(in0, in1);
        default: {
            throw std::runtime_error("Incorrect type of Logical operation");
        }
    }
}

std::shared_ptr<ov::Node> makeLogical(const ov::ParameterVector& inputs,
                                          ov::helpers::LogicalTypes logicalType) {
    switch (logicalType) {
        case ov::helpers::LogicalTypes::LOGICAL_AND:
            return std::make_shared<ov::opset3::LogicalAnd>(inputs[0], inputs[1]);
        case ov::helpers::LogicalTypes::LOGICAL_OR:
            return std::make_shared<ov::opset3::LogicalOr>(inputs[0], inputs[1]);
        case ov::helpers::LogicalTypes::LOGICAL_NOT:
            return std::make_shared<ov::opset3::LogicalNot>(inputs[0]);
        case ov::helpers::LogicalTypes::LOGICAL_XOR:
            return std::make_shared<ov::opset3::LogicalXor>(inputs[0], inputs[1]);
        default: {
            throw std::runtime_error("Incorrect type of Logical operation");
        }
    }
}

}  // namespace builder
}  // namespace ov
