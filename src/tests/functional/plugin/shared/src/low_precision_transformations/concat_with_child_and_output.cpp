// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_child_and_output.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_models/builders.hpp"
#include "ov_lpt_models/concat.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace LayerTestsDefinitions {

std::string ConcatWithChildAndOutputTransformation::getTestCaseName(const testing::TestParamInfo<ConcatWithChildAndOutputTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ConcatWithChildAndOutputTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
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
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    ConcatWithChildAndOutputTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = this->GetParam();

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithChildAndOutput(
        netPrecision, inputShapes, param.fqOnData1, param.fqOnData2);
}

TEST_P(ConcatWithChildAndOutputTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
