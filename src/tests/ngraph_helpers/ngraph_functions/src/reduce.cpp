// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeReduce(const ngraph::Output<Node>& data,
                                         const ngraph::Output<Node>& axes,
                                         bool keepDims,
                                         ngraph::helpers::ReductionType reductionType) {
    switch (reductionType) {
        case helpers::Mean:
            return std::make_shared<ngraph::opset4::ReduceMean>(data, axes, keepDims);
        case helpers::Max:
            return std::make_shared<ngraph::opset4::ReduceMax>(data, axes, keepDims);
        case helpers::Min:
            return std::make_shared<ngraph::opset4::ReduceMin>(data, axes, keepDims);
        case helpers::Prod:
            return std::make_shared<ngraph::opset4::ReduceProd>(data, axes, keepDims);
        case helpers::Sum:
            return std::make_shared<ngraph::opset4::ReduceSum>(data, axes, keepDims);
        case helpers::LogicalOr:
            return std::make_shared<ngraph::opset4::ReduceLogicalOr>(data, axes, keepDims);
        case helpers::LogicalAnd:
            return std::make_shared<ngraph::opset4::ReduceLogicalAnd>(data, axes, keepDims);
        case helpers::L1:
            return std::make_shared<ngraph::opset4::ReduceL1>(data, axes, keepDims);
        case helpers::L2:
            return std::make_shared<ngraph::opset4::ReduceL2>(data, axes, keepDims);
        default:
            throw std::runtime_error("Can't create layer for this reduction type");
    }
}
}  // namespace builder
}  // namespace ngraph
