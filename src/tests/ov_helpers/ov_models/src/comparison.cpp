// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/opsets/opset3.hpp>
#include "ov_models/utils/ov_helpers.hpp"

namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeComparison(const ov::Output<Node> &in0,
                                         const ov::Output<Node> &in1,
                                         ov::helpers::ComparisonTypes comparisonType) {
    switch (comparisonType) {
        case ov::helpers::ComparisonTypes::EQUAL:
            return std::make_shared<ov::opset3::Equal>(in0, in1);
        case ov::helpers::ComparisonTypes::NOT_EQUAL:
            return std::make_shared<ov::opset3::NotEqual>(in0, in1);
        case ov::helpers::ComparisonTypes::GREATER:
            return std::make_shared<ov::opset3::Greater>(in0, in1);
        case ov::helpers::ComparisonTypes::GREATER_EQUAL:
            return std::make_shared<ov::opset3::GreaterEqual>(in0, in1);
        case ov::helpers::ComparisonTypes::IS_FINITE:
            return std::make_shared<ov::op::v10::IsFinite>(in0);
        case ov::helpers::ComparisonTypes::IS_INF:
            return std::make_shared<ov::op::v10::IsInf>(in0);
        case ov::helpers::ComparisonTypes::IS_NAN:
            return std::make_shared<ov::op::v10::IsNaN>(in0);
        case ov::helpers::ComparisonTypes::LESS:
            return std::make_shared<ov::opset3::Less>(in0, in1);
        case ov::helpers::ComparisonTypes::LESS_EQUAL:
            return std::make_shared<ov::opset3::LessEqual>(in0, in1);
        default: {
            throw std::runtime_error("Incorrect type of Comparison operation");
        }
    }
}

}  // namespace builder
}  // namespace ov
