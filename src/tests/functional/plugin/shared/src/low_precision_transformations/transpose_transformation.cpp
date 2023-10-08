// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transpose_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_lpt_models/transpose.hpp"

namespace LayerTestsDefinitions {

std::string TransposeTransformation::getTestCaseName(const testing::TestParamInfo<TransposeTransformationParams>& obj) {
    ngraph::element::Type precision;
    std::string targetDevice;
    TransposeTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.inputShape;

    return result.str();
}

void TransposeTransformation::SetUp() {
    ngraph::element::Type precision;
    TransposeTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::TransposeFunction::getOriginal(
        testValues.inputShape,
        testValues.transposeConstValues,
        testValues.precisionBeforeFq,
        testValues.fqOnData);
}

TEST_P(TransposeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
