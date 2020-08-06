// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_convert_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/fuse_convert_function.hpp"

namespace LayerTestsDefinitions {

std::string FuseConvertTransformation::getTestCaseName(testing::TestParamInfo<FuseConvertTransformationParams> obj) {
    ngraph::element::Type precision;
    std::string targetDevice;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    FuseConvertTransformationTestValues testValues;
    std::tie(precision, targetDevice, version, testValues) = obj.param;

    std::ostringstream result;
    result << version << "_" <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.inputShape << "_" <<
        testValues.fqOnData;

    return result.str();
}

void FuseConvertTransformation::SetUp() {
    ngraph::element::Type precision;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    FuseConvertTransformationTestValues testValues;
    std::tie(precision, targetDevice, version, testValues) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::FuseConvertFunction::getOriginal(
        testValues.inputShape,
        testValues.transposeConstValues,
        testValues.precisionBeforeFq,
        testValues.fqOnData);
}

TEST_P(FuseConvertTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
