// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reshape_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include "ngraph_functions/builders.hpp"
#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/reshape_function.hpp"


namespace LayerTestsDefinitions {

std::string ReshapeTransformation::getTestCaseName(testing::TestParamInfo<ReshapeTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ReshapeTransformationParam param;
    std::tie(netPrecision, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << targetDevice << "_" << toString(params) <<
        "_" << param.inputShape << "_" << param.fakeQuantize << "_{";
    for (size_t i = 0; i < param.reshapeConstValues.size(); ++i) {
        result << param.reshapeConstValues[i];
        if (i != (param.reshapeConstValues.size() - 1ul)) {
            result << ", ";
        }
    }
    result << " }";
    return result.str();
}

void ReshapeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ReshapeTransformationParam param;
    std::tie(netPrecision, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::ReshapeFunction::getOriginal(
        param.inputShape,
        param.reshapeConstValues,
        netPrecision,
        param.fakeQuantize);
}

TEST_P(ReshapeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
