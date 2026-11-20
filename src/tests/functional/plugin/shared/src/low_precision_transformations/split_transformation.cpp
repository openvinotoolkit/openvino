// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/split_transformation.hpp"

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "low_precision/split.hpp"
#include "ov_lpt_models/split.hpp"

namespace LayerTestsDefinitions {
std::string SplitTransformation::getTestCaseName(const testing::TestParamInfo<SplitTransformationParams>& obj) {
    auto [netPrecision, inputShapes, device, param] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShapes, device) << "_" <<
           param.fakeQuantize << "_axis=" << param.splitedAxis << "_n_splits=" << param.numSplit;
    auto test_case_name = result.str();
    std::replace(test_case_name.begin(), test_case_name.end(), '-', '_');
    return test_case_name;
}


void SplitTransformation::SetUp() {
    auto [precision, inputShape, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::SplitFunction::getOriginal(
        precision,
        inputShape,
        param.fakeQuantize,
        param.splitedAxis,
        param.numSplit);
}

TEST_P(SplitTransformation, CompareWithRefImpl) {
    run();
};
} // namespace LayerTestsDefinitions
