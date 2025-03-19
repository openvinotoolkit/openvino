// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_child_and_output.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatWithChildAndOutputTransformation::getTestCaseName(const testing::TestParamInfo<ConcatWithChildAndOutputTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShapes;
    std::string targetDevice;
    ConcatWithChildAndOutputTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = obj.param;

    std::ostringstream result;
    result <<
           get_test_case_name_by_params(netPrecision, inputShapes, targetDevice, params) <<
           param.fqOnData1 << param.fqOnData2;

    return result.str();
}

/*
*     FQ  FQ
*      \ /
*     concat
*      /  \
*   clamp  \
*    /      \
* output1  output2
*/

void ConcatWithChildAndOutputTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShapes;
    ConcatWithChildAndOutputTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = this->GetParam();

    init_input_shapes({ inputShapes, inputShapes });

    function = ov::builder::subgraph::ConcatFunction::getOriginalWithChildAndOutput(
        netPrecision, inputShapes, param.fqOnData1, param.fqOnData2);
}

TEST_P(ConcatWithChildAndOutputTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
