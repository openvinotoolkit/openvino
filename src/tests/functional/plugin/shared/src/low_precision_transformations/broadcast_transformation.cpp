// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/broadcast_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "lpt_ngraph_functions/broadcast_function.hpp"

namespace LayerTestsDefinitions {
std::string BroadcastTransformation::getTestCaseName(const testing::TestParamInfo<BroadcastTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    BroadcastTransformationParam param;
    size_t opset;
    std::string mode;
    std::tie(netPrecision, inputShape, targetDevice, params, param, opset, mode) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
           param.fakeQuantize << "_" << opset << "_" << mode;
    return result.str();
}

void BroadcastTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    BroadcastTransformationParam param;
    size_t opset;
    std::string mode;
    std::tie(netPrecision, inputShape, targetDevice, params, param, opset, mode) = GetParam();

    function = ngraph::builder::subgraph::BroadcastFunction::getOriginal(
            netPrecision,
            inputShape,
            param.fakeQuantize,
            opset,
            mode);
}

TEST_P(BroadcastTransformation, CompareWithRefImpl) {
    Run();
}
} // namespace LayerTestsDefinitions
