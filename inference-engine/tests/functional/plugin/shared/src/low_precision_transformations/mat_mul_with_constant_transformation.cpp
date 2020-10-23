// Copyright (C) 2020 Intel Corporation
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
#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"

namespace LayerTestsDefinitions {

std::string MatMulWithConstantTransformation::getTestCaseName(testing::TestParamInfo<MatMulWithConstantTransformationParams> obj) {
    ngraph::element::Type precision;
    std::string targetDevice;
    MatMulWithConstantTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fqOnData << "_" <<
        testValues.fqOnWeights;

    return result.str();
}

InferenceEngine::Blob::Ptr MatMulWithConstantTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    if ((info.name() != "input1") && (info.name() != "input2")) {
        THROW_IE_EXCEPTION << "unexpected layer name " << info.name();
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
        THROW_IE_EXCEPTION << "unexpected input name " << info.name();
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
        testValues.weightsConstShape,
        testValues.weightsConstValues,
        testValues.fqOnWeights);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(MatMulWithConstantTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
