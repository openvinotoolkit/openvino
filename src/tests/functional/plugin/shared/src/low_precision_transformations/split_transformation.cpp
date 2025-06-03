// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/split_transformation.hpp"

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
    return result.str();
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
