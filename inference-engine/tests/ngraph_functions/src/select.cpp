// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "ngraph_functions/select.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace NGraphFunctions {

    Select::Select(ngraph::element::Type inType, const std::vector<std::vector<size_t>> &inputShapes, ngraph::op::AutoBroadcastSpec broadcast) {
        ngraph::ParameterVector paramNodesVector;

        auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::boolean, ngraph::Shape(inputShapes[CONDITION]));
        paramNodesVector.push_back(paramNode);
        for (size_t i = 1; i < inputShapes.size(); i++) {
            paramNode = std::make_shared<ngraph::opset1::Parameter>(inType, ngraph::Shape(inputShapes[i]));
            paramNodesVector.push_back(paramNode);
        }
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramNodesVector));
        broadcastType = broadcast;

        auto SelectNode = std::make_shared<ngraph::opset1::Select>(paramOuts[CONDITION], paramOuts[THEN], paramOuts[ELSE], broadcastType);

        auto result = std::make_shared<ngraph::opset1::Result>(SelectNode);

        fnPtr = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, paramNodesVector, "select");
    }

}  // namespace NGraphFunctions