// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_different_precision_on_children.hpp"

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

std::string ConcatWithDifferentChildrenTransformation::getTestCaseName(const testing::TestParamInfo<ConcatWithDifferentChildrenTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildrenTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
        "_axis_" << param.axis << "_" << param.fqOnData1 << param.fqOnData2;

    return result.str();
}

InferenceEngine::Blob::Ptr ConcatWithDifferentChildrenTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildrenTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(ngraph::element::u8, info.getTensorDesc(), k);
}

void ConcatWithDifferentChildrenTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    ConcatWithDifferentChildrenTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = this->GetParam();

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithDifferentPrecisionOnChildren(
        netPrecision, inputShapes, param.axis, param.fqOnData1, param.fqOnData2);
}

TEST_P(ConcatWithDifferentChildrenTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
