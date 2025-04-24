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
    ov::element::Type netPrecision;
    ov::PartialShape inputShapes;
    std::string targetDevice;
    ConcatWithSplitTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShapes, targetDevice, params) << param.fqOnData1 << "_" << param.fqOnData2;
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
    ov::element::Type netPrecision;
    ov::PartialShape inputShapes;
    ConcatWithSplitTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = this->GetParam();

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
