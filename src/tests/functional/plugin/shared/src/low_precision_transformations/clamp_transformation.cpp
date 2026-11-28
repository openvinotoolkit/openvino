// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/clamp_transformation.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/clamp.hpp"

namespace LayerTestsDefinitions {

std::string ClampTransformation::getTestCaseName(const testing::TestParamInfo<ClampTransformationParams>& obj) {
    auto [netPrecision, inputShape, targetDevice, param] = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice) << "_" <<
           param.fakeQuantize << "_" <<
        "min=" << param.clampLowConst <<
        "max=" << param.clampHighConst;
    auto test_case_name = result.str();
    std::replace(test_case_name.begin(), test_case_name.end(), '-', '_');
    return test_case_name;
}

void ClampTransformation::SetUp() {
    auto [netPrecision, inputShape, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::ClampFunction::getOriginal(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        param.clampLowConst,
        param.clampHighConst);
}

TEST_P(ClampTransformation, CompareWithRefImpl) {
    run();
};

} // namespace LayerTestsDefinitions
