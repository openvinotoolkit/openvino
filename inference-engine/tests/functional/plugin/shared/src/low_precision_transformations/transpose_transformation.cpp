// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transpose_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/transpose_function.hpp"

namespace LayerTestsDefinitions {

std::string TransposeTransformation::getTestCaseName(testing::TestParamInfo<TransposeTransformationParams> obj) {
    ngraph::element::Type precision;
    std::string targetDevice;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    TransposeTransformationTestValues testValues;
    std::tie(precision, targetDevice, version, testValues) = obj.param;

    std::ostringstream result;
    result << version << "_" <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.inputShape;

    return result.str();
}

void TransposeTransformation::SetUp() {
    ngraph::element::Type precision;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    TransposeTransformationTestValues testValues;
    std::tie(precision, targetDevice, version, testValues) = this->GetParam();

    ConfigurePlugin(version);

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
