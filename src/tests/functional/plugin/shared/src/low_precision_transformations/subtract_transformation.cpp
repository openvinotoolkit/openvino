// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/subtract_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/subtract.hpp"



namespace LayerTestsDefinitions {

std::string SubtractTransformation::getTestCaseName(const testing::TestParamInfo<SubtractTransformationParams>& obj) {
    auto [netPrecision, inputShapes, device] = obj.param;
    return get_test_case_name_by_params(netPrecision, inputShapes, device);
}

void SubtractTransformation::SetUp() {
    auto [netPrecision, inputShape, device] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::SubtractFunction::getOriginal(netPrecision, inputShape);
}

TEST_P(SubtractTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
