// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_different_precision_on_childs.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/low_precision_transformations/concat_function.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace LayerTestsDefinitions {

std::string ConcatWithDifferentChildsTransformation::getTestCaseName(testing::TestParamInfo<ConcatWithDifferentChildsTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildsTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, multiChannel) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
        (multiChannel ? "_multichannel" : "") << param.fqOnData1 << param.fqOnData2;

    return result.str();
}

InferenceEngine::Blob::Ptr ConcatWithDifferentChildsTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildsTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, multiChannel) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(params.precisionsOnActivations[0], info.getTensorDesc(), k);
}

void ConcatWithDifferentChildsTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    ConcatWithDifferentChildsTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, multiChannel) = this->GetParam();

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithDifferentPrecisionOnChilds(
        netPrecision, inputShapes, param.fqOnData1, param.fqOnData2);
}

TEST_P(ConcatWithDifferentChildsTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
