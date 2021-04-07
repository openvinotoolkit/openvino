// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reduce_sum_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "lpt_ngraph_functions/reduce_function.hpp"

namespace LayerTestsDefinitions {

std::string ReduceSumTransformation::getTestCaseName(testing::TestParamInfo<ReduceSumTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ReduceSumTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
        param.fakeQuantize << (param.keepDims ? "_keepDims_" : "") << "_reduce_axis_";
    for (const auto& elem : param.constantValues) {
        result << elem << "_";
    }

    return result.str();
}

void ReduceSumTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ReduceSumTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = GetParam();

    function = ngraph::builder::subgraph::ReduceFunction::getOriginal<ngraph::opset1::ReduceSum>(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        param.constantValues,
        param.keepDims);
}

void ReduceSumTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = getRuntimePrecision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(ReduceSumTransformation, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
