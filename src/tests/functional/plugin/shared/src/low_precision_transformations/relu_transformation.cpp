// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/relu_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/relu.hpp"

namespace LayerTestsDefinitions {

std::string ReluTransformation::getTestCaseName(const testing::TestParamInfo<ReluTransformationParams>& obj) {
    const auto& [precision, inputShape, targetDevice, testValues] = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fakeQuantize;

    return result.str();
}


void ReluTransformation::SetUp() {
    const auto& [precision, inputShape, _targetDevice, testValues] = this->GetParam();
    targetDevice = _targetDevice;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::ReluFunction::getOriginal(inputShape, precision, testValues.fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(ReluTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
