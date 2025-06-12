// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mvn_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"

#include "ov_lpt_models/mvn.hpp"

namespace LayerTestsDefinitions {

std::string MVNTransformation::getTestCaseName(const testing::TestParamInfo<MVNTransformationParams>& obj) {
    auto [precision, shape, device, reductionAxes, normalizeVariance] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(precision, shape, device) <<
           "_" << reductionAxes << "_" << normalizeVariance;
    return result.str();
}

void MVNTransformation::SetUp() {
    auto [precision, shape, device, reductionAxes, normalizeVariance] = this->GetParam();
    targetDevice = device;

    init_input_shapes(shape);

    function = ov::builder::subgraph::MVNFunction::getOriginal(
        precision,
        shape,
        reductionAxes,
        normalizeVariance);
}

TEST_P(MVNTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
