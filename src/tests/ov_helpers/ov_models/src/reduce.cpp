// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {
std::shared_ptr<ov::Node> makeReduce(const ov::Output<Node>& data,
                                         const ov::Output<Node>& axes,
                                         bool keepDims,
                                         ov::helpers::ReductionType reductionType) {
    switch (reductionType) {
        case ov::helpers::Mean:
            return std::make_shared<ov::opset4::ReduceMean>(data, axes, keepDims);
        case ov::helpers::Max:
            return std::make_shared<ov::opset4::ReduceMax>(data, axes, keepDims);
        case ov::helpers::Min:
            return std::make_shared<ov::opset4::ReduceMin>(data, axes, keepDims);
        case ov::helpers::Prod:
            return std::make_shared<ov::opset4::ReduceProd>(data, axes, keepDims);
        case ov::helpers::Sum:
            return std::make_shared<ov::opset4::ReduceSum>(data, axes, keepDims);
        case ov::helpers::LogicalOr:
            return std::make_shared<ov::opset4::ReduceLogicalOr>(data, axes, keepDims);
        case ov::helpers::LogicalAnd:
            return std::make_shared<ov::opset4::ReduceLogicalAnd>(data, axes, keepDims);
        case ov::helpers::L1:
            return std::make_shared<ov::opset4::ReduceL1>(data, axes, keepDims);
        case ov::helpers::L2:
            return std::make_shared<ov::opset4::ReduceL2>(data, axes, keepDims);
        default:
            throw std::runtime_error("Can't create layer for this reduction type");
    }
}
}  // namespace builder
}  // namespace ov
