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
#include "ov_lpt_models/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::string MatMulTransformation::getTestCaseName(const testing::TestParamInfo<MatMulTransformationParams>& obj) {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    MatMulTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.inputShape1 << "_" <<
        testValues.fqOnData1 << "_" <<
        testValues.inputShape2 << "_" <<
        testValues.fqOnData2 << "_" <<
        testValues.expectedRuntimePrecision << "_" <<
        testValues.expectedKernelName;

    return result.str();
}


void MatMulTransformation::SetUp() {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    MatMulTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    init_input_shapes({ testValues.inputShape1, testValues.inputShape2 });

    function = ov::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        testValues.inputShape1,
        testValues.fqOnData1,
        testValues.inputShape2,
        testValues.fqOnData2);

    ov::pass::InitNodeInfo().run_on_model(function);
}

void MatMulTransformation::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LayerTransformation::run();

    const auto params = std::get<3>(GetParam());
    const auto actualType = get_runtime_precision(params.expectedKernelName);

    EXPECT_EQ(actualType, params.expectedRuntimePrecision);
}

TEST_P(MatMulTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
