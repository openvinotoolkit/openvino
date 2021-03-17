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
#include "lpt_ngraph_functions/multiply_with_one_parent_function.hpp"

namespace LayerTestsDefinitions {

std::string MultiplyWithOneParentTransformation::getTestCaseName(testing::TestParamInfo<MultiplyWithOneParentTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    MultiplyWithOneParentTransformationValues values;

    std::tie(netPrecision, inputShape, targetDevice, values) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << CommonTestUtils::vec2str(inputShape);
    return result.str();
}

void MultiplyWithOneParentTransformation::SetUp() {
    threshold = 0.01f;

    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    MultiplyWithOneParentTransformationValues values;
    std::tie(netPrecision, inputShape, targetDevice, values) = this->GetParam();

    function = ngraph::builder::subgraph::MultiplyWithOneParentFunction::getOriginal(netPrecision, inputShape, values.fakeQuantize);
}

TEST_P(MultiplyWithOneParentTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
