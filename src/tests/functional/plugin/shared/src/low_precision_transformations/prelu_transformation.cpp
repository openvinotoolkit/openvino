// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/prelu_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/prelu.hpp"

namespace LayerTestsDefinitions {

std::string PReluTransformation::getTestCaseName(const testing::TestParamInfo<PReluTransformationParams>& obj) {
    const auto& [precision, inputShape, targetDevice, testValues] = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fakeQuantize;

    return result.str();
}


void PReluTransformation::SetUp() {
    const auto& [precision, inputShape, _targetDevice, testValues] = this->GetParam();
    targetDevice = _targetDevice;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::PReluFunction::getOriginal(inputShape, precision, testValues.fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(PReluTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions

