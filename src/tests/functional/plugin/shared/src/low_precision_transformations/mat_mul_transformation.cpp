// Copyright (C) 2018-2024 Intel Corporation
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
        "IN1=" << testValues.inputShape1 << "_" <<
        testValues.fqOnData1 << "_" <<
        "IN2=" << testValues.inputShape2 << "_" <<
        testValues.fqOnData2 << "_" <<
        testValues.requantization;

    return result.str();
}


void MatMulTransformation::SetUp() {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    MatMulTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    init_input_shapes({ testValues.inputShape1, testValues.inputShape2 });

    function = ov::builder::subgraph::MatMulFunction::getOriginal(
        ov::element::f16, // precision,
        testValues.inputShape1,
        testValues.fqOnData1,
        testValues.inputShape2,
        testValues.fqOnData2,
        testValues.requantization);

    ov::pass::InitNodeInfo().run_on_model(function);

    ov::pass::Serialize("/Users/eshoguli/projects/openvino/test.original.xml", "/Users/eshoguli/projects/openvino/test.original.bin").run_on_model(function);
}

void MatMulTransformation::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LayerTransformation::run();

    const auto& actualType = get_runtime_precision_by_type("MatMul");
    const auto expected = std::get<3>(GetParam());
    EXPECT_EQ(expected.expectedRuntimePrecision, actualType);

    const auto& actualPrimitiveType = get_property_by_type("MatMul", "primitiveType");
    const auto expectedPrimitiveType = "gemm_acl_i8";
    EXPECT_EQ(expectedPrimitiveType, actualPrimitiveType);
}

TEST_P(MatMulTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
