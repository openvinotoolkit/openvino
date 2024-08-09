// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fully_connected_transformation.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>


#include "common_test_utils/common_utils.hpp"
#include "openvino/util/common_util.hpp"
#include "ov_lpt_models/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::string FullyConnectedTransformation::getTestCaseName(const testing::TestParamInfo<FullyConnectedTransformationParams>& obj) {
    ov::element::Type precision;
    MatMulShapes shapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ov::element::Type weightsType;
    Activation activation;
    std::string expectedPrimitiveType;
    std::tie(precision, shapes, targetDevice, params, weightsType, activation, expectedPrimitiveType) = obj.param;

    std::ostringstream result;
    result <<
        get_test_case_name_by_params(precision, shapes.inputA, targetDevice, params) <<
        shapes.inputB << "_" <<
        shapes.transposeA << "_" <<
        shapes.transposeB << "_" <<
        weightsType << "_" <<
        activation.exist << "_" <<
        activation.originalLayersNames << "_" <<
        expectedPrimitiveType;

    return result.str();
}

void FullyConnectedTransformation::SetUp() {
    ov::element::Type precision;
    MatMulShapes shapes;
    ov::pass::low_precision::LayerTransformation::Params params;
    ov::element::Type weightsType;
    Activation activation;
    std::string expectedPrimitiveType;
    std::tie(precision, shapes, targetDevice, params, weightsType, activation, expectedPrimitiveType) = this->GetParam();

    init_input_shapes({ shapes.inputA, shapes.inputB });

    function = ov::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        shapes.inputA,
        shapes.inputB,
        shapes.transposeA,
        shapes.transposeB,
        weightsType == ov::element::i8,
        activation.exist);

    ov::pass::Serialize(
            "/Users/eshoguli/projects/openvino_matmul/test.original.xml",
            "/Users/eshoguli/projects/openvino_matmul/test.original.bin").run_on_model(function);
}

TEST_P(FullyConnectedTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();

    const auto& activation = std::get<5>(GetParam());
    const auto originalLayersNames = get_property_by_type("FullyConnected", "originalLayersNames");
    EXPECT_EQ(ov::util::to_lower(activation.originalLayersNames), originalLayersNames);

    const auto& actualPrecision = get_runtime_precision_by_type("FullyConnected");
    const auto expectedPrecision = std::get<4>(GetParam());
    EXPECT_EQ(actualPrecision, expectedPrecision.to_string());

    const auto& expectedPrimitiveType = std::get<6>(GetParam());
    const std::string actualPrimitiveType = get_property_by_type("FullyConnected", "primitiveType");
    EXPECT_EQ(expectedPrimitiveType, actualPrimitiveType);
};

}  // namespace LayerTestsDefinitions
