// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_split_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatWithSplitTransformation::getTestCaseName(const testing::TestParamInfo<ConcatWithSplitTransformationParams>& obj) {
    auto [netPrecision, inputShapes, device, param] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShapes, device) << param.fqOnData1 << "_" << param.fqOnData2;
    return result.str();
}


/*
* FQ       FQ
*  \       /
*   \    Split
*    \   /   \
*   Concat  Convolution
*/

void ConcatWithSplitTransformation::SetUp() {
    auto [netPrecision, inputShapes, device, param] = this->GetParam();
    targetDevice = device;

    auto inputShape1 = inputShapes;
    const size_t numSplit = 2;
    inputShape1[1] = inputShape1[1].get_length() / numSplit;
    init_input_shapes({ inputShape1, inputShapes });

    function = ov::builder::subgraph::ConcatFunction::getOriginalWithSplitedIntermediate(
        netPrecision,
        inputShapes,
        param.fqOnData1,
        param.fqOnData2,
        true);
}

TEST_P(ConcatWithSplitTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
