// Copyright (C) 2019 Intel Corporation
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
            return std::make_shared<ngraph::opset3::ReduceMean>(data, axes, keepDims);
        case helpers::Max:
            return std::make_shared<ngraph::opset3::ReduceMax>(data, axes, keepDims);
        case helpers::Min:
            return std::make_shared<ngraph::opset3::ReduceMin>(data, axes, keepDims);
        case helpers::Prod:
            return std::make_shared<ngraph::opset3::ReduceProd>(data, axes, keepDims);
        case helpers::Sum:
            return std::make_shared<ngraph::opset3::ReduceSum>(data, axes, keepDims);
        case helpers::LogicalOr:
            return std::make_shared<ngraph::opset3::LogicalOr>(data, axes);
        case helpers::LogicalAnd:
            return std::make_shared<ngraph::opset3::LogicalAnd>(data, axes);
        case helpers::LogicalXor:
            return std::make_shared<ngraph::opset3::LogicalXor>(data, axes);
        default:
            throw std::runtime_error("Can't create layer for this reduction type");
    }
}
}  // namespace builder
}  // namespace ngraph
