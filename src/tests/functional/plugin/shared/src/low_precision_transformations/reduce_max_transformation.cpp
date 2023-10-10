// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reduce_max_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "ov_lpt_models/reduce.hpp"

namespace LayerTestsDefinitions {

std::string ReduceMaxTransformation::getTestCaseName(const testing::TestParamInfo<ReduceMaxTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ReduceMaxTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
        param.fakeQuantize << (param.keepDims ? "_keepDims_" : "") << "_reduce_axis_";
    for (const auto& elem : param.constantValues) {
        result << elem << "_";
    }

    return result.str();
}

void ReduceMaxTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ReduceMaxTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = GetParam();

    ngraph::builder::subgraph::DequantizationOperations::Convert convert;
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;

    function = ngraph::builder::subgraph::ReduceFunction::get<ngraph::opset1::ReduceMax>(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        convert,
        dequantizationBefore,
        param.constantValues,
        param.keepDims,
        dequantizationAfter);
}

void ReduceMaxTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = getRuntimePrecision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(ReduceMaxTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

} // namespace LayerTestsDefinitions
