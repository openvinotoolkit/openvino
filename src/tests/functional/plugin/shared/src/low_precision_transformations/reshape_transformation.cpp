// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reshape_transformation.hpp"

#include <memory>
#include <tuple>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/reshape.hpp"

namespace LayerTestsDefinitions {

std::string ReshapeTransformation::getTestCaseName(const testing::TestParamInfo<ReshapeTransformationParams>& obj) {
    auto [netPrecision, device, param] = obj.param;
    std::ostringstream result;
    result << netPrecision << "_" << device <<
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
    auto [netPrecision, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(param.inputShape);

    function = ov::builder::subgraph::ReshapeFunction::getOriginal(
        param.inputShape,
        param.reshapeConstValues,
        netPrecision,
        param.fakeQuantize);
}

void ReshapeTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<2>(GetParam());

    EXPECT_TRUE(check_execution_order(params.executionOrder));

    auto actualPrecision = get_runtime_precision_by_type(params.layerType);
    const auto expectedPrecision = params.expectedKernelType;
    if ((expectedPrecision == "f32") && (actualPrecision == "f16")) {
        actualPrecision = "f32";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(ReshapeTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
