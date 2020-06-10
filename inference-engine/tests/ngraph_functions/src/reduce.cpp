// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeReduce(std::vector<ngraph::Output<Node>> &in,
                                         const std::vector<int> &reductionAxes,
                                         bool keepDims,
                                         ngraph::helpers::ReductionType reductionType) {
    auto reductionAxesNode = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64,
                                                                        ngraph::Shape({reductionAxes.size()}),
                                                                        reductionAxes);
    switch (reductionType) {
        case helpers::Mean:
            return std::make_shared<ngraph::opset3::ReduceMean>(in.at(0), reductionAxesNode, keepDims);
        case helpers::Max:
            return std::make_shared<ngraph::opset3::ReduceMax>(in.at(0), reductionAxesNode, keepDims);
        case helpers::Min:
            return std::make_shared<ngraph::opset3::ReduceMin>(in.at(0), reductionAxesNode, keepDims);
        case helpers::Prod:
            return std::make_shared<ngraph::opset3::ReduceProd>(in.at(0), reductionAxesNode, keepDims);
        case helpers::Sum:
            return std::make_shared<ngraph::opset3::ReduceSum>(in.at(0), reductionAxesNode, keepDims);
        case helpers::LogicalOr:
            return std::make_shared<ngraph::opset3::LogicalOr>(in.at(0), reductionAxesNode);
        case helpers::LogicalAnd:
            return std::make_shared<ngraph::opset3::LogicalAnd>(in.at(0), reductionAxesNode);
        case helpers::LogicalXor:
            return std::make_shared<ngraph::opset3::LogicalXor>(in.at(0), reductionAxesNode);
    }
}
}  // namespace builder
}  // namespace ngraph
