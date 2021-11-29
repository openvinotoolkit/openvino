// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_different_precision_on_children.hpp"

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

std::string ConcatWithDifferentChildrenTransformation::getTestCaseName(testing::TestParamInfo<ConcatWithDifferentChildrenTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildrenTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, multiChannel) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
        (multiChannel ? "_multichannel" : "") << param.fqOnData1 << param.fqOnData2;

    return result.str();
}

InferenceEngine::Blob::Ptr ConcatWithDifferentChildrenTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildrenTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, multiChannel) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(params.precisionsOnActivations[0], info.getTensorDesc(), k);
}

void ConcatWithDifferentChildrenTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    ConcatWithDifferentChildrenTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, multiChannel) = this->GetParam();

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithDifferentPrecisionOnChildren(
        netPrecision, inputShapes, param.fqOnData1, param.fqOnData2);

    validate();
}

void ConcatWithDifferentChildrenTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildrenTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, multiChannel) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    ASSERT_EQ(2ul, transformed->get_output_size());
    for (size_t i = 0; i < 2ul; ++i) {
        const auto output = transformed->get_output_op(0);
        const auto scaleShift = output->get_input_node_shared_ptr(0);
        const std::string typeName = scaleShift->get_type_name();
        ASSERT_EQ("ScaleShiftIE", typeName);
    }
}

TEST_P(ConcatWithDifferentChildrenTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
