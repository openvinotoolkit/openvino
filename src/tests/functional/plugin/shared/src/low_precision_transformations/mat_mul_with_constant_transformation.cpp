// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <vector>
#include <string>
#include <queue>

#include "transformations/init_node_info.hpp"
#include "low_precision_transformations/mat_mul_transformation.hpp"
#include "low_precision_transformations/mat_mul_with_constant_transformation.hpp"
#include "ov_lpt_models/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::string MatMulWithConstantTransformation::getTestCaseName(const testing::TestParamInfo<MatMulWithConstantTransformationParams>& obj) {
    ov::element::Type precision;
    std::string targetDevice;
    MatMulWithConstantTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        testValues.inputShape << "_" <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fqOnData << "_" <<
        testValues.fqOnWeights << "_" <<
        testValues.deqOnWeights;

    return result.str();
}


void MatMulWithConstantTransformation::SetUp() {
    ov::element::Type precision;
    MatMulWithConstantTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = this->GetParam();

    init_input_shapes(testValues.inputShape);

    function = ov::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        testValues.inputShape,
        testValues.fqOnData,
        testValues.weights,
        testValues.fqOnWeights,
        testValues.deqOnWeights);

    ov::pass::InitNodeInfo().run_on_model(function);
}

void MatMulWithConstantTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<2>(GetParam());
    const auto actualPrecision = get_runtime_precision_by_type(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ov::element::f16) {
        expectedPrecision = "FP16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(MatMulWithConstantTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
