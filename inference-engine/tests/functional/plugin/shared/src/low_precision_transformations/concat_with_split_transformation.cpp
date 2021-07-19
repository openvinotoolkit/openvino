// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_split_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/builders.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace LayerTestsDefinitions {

std::string ConcatWithSplitTransformation::getTestCaseName(testing::TestParamInfo<ConcatWithSplitTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ConcatWithSplitTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) << param.fqOnData1 << "_" << param.fqOnData2;
    return result.str();
}

InferenceEngine::Blob::Ptr ConcatWithSplitTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ConcatWithSplitTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(ngraph::element::u8, info.getTensorDesc(), k);
}

/*
* FQ       FQ
*  \       /
*   \    Split
*    \   /   \
*   Concat  Convolution
*/

void ConcatWithSplitTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    ConcatWithSplitTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = this->GetParam();

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithSplitedIntermediate(
        netPrecision,
        inputShapes,
        param.fqOnData1,
        param.fqOnData2,
        true);
}

TEST_P(ConcatWithSplitTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
