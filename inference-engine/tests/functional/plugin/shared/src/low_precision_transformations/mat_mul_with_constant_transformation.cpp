// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mat_mul_with_constant_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <queue>
#include <ie_core.hpp>

#include "ngraph/op/op.hpp"
#include <transformations/init_node_info.hpp>
#include "low_precision_transformations/mat_mul_transformation.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/mat_mul_function.hpp"

namespace LayerTestsDefinitions {

std::string MatMulWithConstantTransformation::getTestCaseName(testing::TestParamInfo<MatMulWithConstantTransformationParams> obj) {
    ngraph::element::Type precision;
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

InferenceEngine::Blob::Ptr MatMulWithConstantTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    if ((info.name() != "input1") && (info.name() != "input2")) {
        IE_THROW() << "unexpected layer name " << info.name();
    }

    size_t low;
    size_t high;
    if (info.name() == "input1") {
        low = 1ul;
        high = 5ul;
    } else if (info.name() == "input2") {
        low = 5ul;
        high = 10ul;
    } else {
        IE_THROW() << "unexpected input name " << info.name();
    }

    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), high - low, low, 1ul);
}

void MatMulWithConstantTransformation::SetUp() {
    ngraph::element::Type precision;
    MatMulWithConstantTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        testValues.inputShape,
        testValues.fqOnData,
        testValues.weights,
        testValues.fqOnWeights,
        testValues.deqOnWeights);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

void MatMulWithConstantTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<2>(GetParam());
    const auto actualPrecision = getRuntimePrecisionByType(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ngraph::element::f16) {
        expectedPrecision = "FP16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(MatMulWithConstantTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
