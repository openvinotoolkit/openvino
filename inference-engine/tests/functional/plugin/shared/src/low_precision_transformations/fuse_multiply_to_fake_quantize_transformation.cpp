// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_multiply_to_fake_quantize_transformation.hpp"

#include <tuple>
#include <sstream>
#include <string>
#include <vector>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/fuse_multiply_to_fake_quantize_function.hpp"

namespace LayerTestsDefinitions {

std::string FuseMultiplyToFakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<FuseMultiplyToFakeQuantizeTransformationParams> obj) {
    std::string targetDevice;
    FuseMultiplyToFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result << targetDevice << "_" <<
        testValues.actual.dequantization << "_" <<
        testValues.actual.fakeQuantizeOnData;
    return result.str();
}

void FuseMultiplyToFakeQuantizeTransformation::SetUp() {
    FuseMultiplyToFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::FuseMultiplyToFakeQuantizeFunction::get(
        testValues.inputShape,
        testValues.actual.fakeQuantizeOnData,
        testValues.actual.dequantization);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(FuseMultiplyToFakeQuantizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
