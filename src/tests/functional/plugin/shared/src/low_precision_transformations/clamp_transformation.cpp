// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/clamp_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/clamp.hpp"

namespace LayerTestsDefinitions {

std::string ClampTransformation::getTestCaseName(const testing::TestParamInfo<ClampTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ClampTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) << "_" <<
           param.fakeQuantize << "_" <<
        "min=" << param.clampLowConst <<
        "max=" << param.clampHighConst;
    return result.str();
}

void ClampTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ClampTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

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
