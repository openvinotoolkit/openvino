// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reduce_max_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/reduce.hpp"

namespace LayerTestsDefinitions {

std::string ReduceMaxTransformation::getTestCaseName(const testing::TestParamInfo<ReduceMaxTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ReduceMaxTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) << "_" <<
           param.fakeQuantize << (param.keepDims ? "_keepDims_" : "") << "_reduce_axis_";
    for (const auto& elem : param.constantValues) {
        result << elem << "_";
    }

    return result.str();
}

void ReduceMaxTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ReduceMaxTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = GetParam();

    init_input_shapes(inputShape);

    ov::builder::subgraph::DequantizationOperations::Convert convert;
    ov::builder::subgraph::DequantizationOperations dequantizationBefore;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter;

    function = ov::builder::subgraph::ReduceFunction::get<ov::op::v1::ReduceMax>(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        convert,
        dequantizationBefore,
        param.constantValues,
        param.keepDims,
        dequantizationAfter);
}

void ReduceMaxTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = get_runtime_precision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(ReduceMaxTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

} // namespace LayerTestsDefinitions
