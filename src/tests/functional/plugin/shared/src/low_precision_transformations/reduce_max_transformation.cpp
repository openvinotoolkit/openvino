// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reduce_max_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/reduce.hpp"
#include "openvino/op/reduce_max.hpp"

namespace LayerTestsDefinitions {

std::string ReduceMaxTransformation::getTestCaseName(const testing::TestParamInfo<ReduceMaxTransformationParams>& obj) {
    auto [netPrecision, inputShape, device, param] = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, device) << "_" <<
           param.fakeQuantize << (param.keepDims ? "_keepDims_" : "") << "_reduce_axis_";
    for (const auto& elem : param.constantValues) {
        result << elem << "_";
    }

    return result.str();
}

void ReduceMaxTransformation::SetUp() {
    auto [netPrecision, inputShape, device, param] = GetParam();
    targetDevice = device;

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

    const auto params = std::get<3>(GetParam());
    const auto actualType = get_runtime_precision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(ReduceMaxTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

} // namespace LayerTestsDefinitions
