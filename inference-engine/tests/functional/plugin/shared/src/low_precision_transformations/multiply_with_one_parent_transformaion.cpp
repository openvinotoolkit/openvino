// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_with_one_parent_transformation.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ie_core.hpp>
#include "common_test_utils/common_utils.hpp"
#include "ngraph_functions/low_precision_transformations/multiply_with_one_parent_function.hpp"

namespace LayerTestsDefinitions {

std::string MultiplyWithOneParentTransformation::getTestCaseName(testing::TestParamInfo<MultiplyWithOneParentTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    MultiplyWithOneParentTransformationValues values;

    std::tie(netPrecision, inputShape, targetDevice, values) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << CommonTestUtils::vec2str(inputShape);
    return result.str();
}

void MultiplyWithOneParentTransformation::SetUp() {
    threshold = 0.01f;

    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    MultiplyWithOneParentTransformationValues values;
    std::tie(netPrecision, inputShape, targetDevice, values) = this->GetParam();
    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    function = ngraph::builder::subgraph::MultiplyWithOneParentFunction::getOriginal(precision, inputShape, values.fakeQuantize);
}

TEST_P(MultiplyWithOneParentTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
