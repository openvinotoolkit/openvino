// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_subtract_to_fake_quantize_transformation.hpp"

#include <tuple>
#include <sstream>
#include <string>
#include <vector>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/fuse_subtract_to_fake_quantize.hpp"

namespace LayerTestsDefinitions {

std::string FuseSubtractToFakeQuantizeTransformation::getTestCaseName(const testing::TestParamInfo<FuseSubtractToFakeQuantizeTransformationParams>& obj) {
    std::string targetDevice;
    FuseSubtractToFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result << targetDevice << "_" <<
        testValues.actual.dequantization << "_" <<
        testValues.actual.fakeQuantizeOnData;
    return result.str();
}

void FuseSubtractToFakeQuantizeTransformation::SetUp() {
    FuseSubtractToFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    init_input_shapes(testValues.inputShape);

    function = ov::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
        testValues.inputShape,
        testValues.actual.fakeQuantizeOnData,
        testValues.actual.dequantization);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(FuseSubtractToFakeQuantizeTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
