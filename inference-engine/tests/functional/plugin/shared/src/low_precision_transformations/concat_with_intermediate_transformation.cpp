// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_intermediate_transformation.hpp"

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

std::string ConcatWithIntermediateTransformation::getTestCaseName(testing::TestParamInfo<ConcatWithIntermediateTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShapes, targetDevice, params, transparentIntermediate, multichannel) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
        (transparentIntermediate ? "" : "_notTransparentIntermediate") <<
        (multichannel ? "_multichannel" : "");

    return result.str();
}

InferenceEngine::Blob::Ptr ConcatWithIntermediateTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params trasformationParams;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShape, targetDevice, trasformationParams, transparentIntermediate, multichannel) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(trasformationParams.precisionsOnActivations[0], info.getTensorDesc(), k);
}

/*
* FQ       FQ
*  \       /
*   \  Intermediate (MaxPooling or Convolution)
*    \  /    \
*   Concat   Convolution
*/

void ConcatWithIntermediateTransformation::SetUp() {
    ngraph::element::Type ngPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params trasformationParams;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(ngPrecision, inputShape, targetDevice, trasformationParams, transparentIntermediate, multichannel) = this->GetParam();

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithIntermediate(
        ngPrecision,
        inputShape,
        transparentIntermediate,
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} });

    validate();
}

void ConcatWithIntermediateTransformation::validate() {
    ngraph::element::Type netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShape, targetDevice, params, transparentIntermediate, multichannel) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));
    ASSERT_EQ(2ul, transformed->get_output_size());

    const auto concatOutput = transformed->get_output_op(0);
    const auto scaleShiftOrConcat = concatOutput->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShiftOrConcat->get_type_name();
    if (transparentIntermediate) {
        ASSERT_EQ("ScaleShiftIE", typeName);
    } else {
        ASSERT_EQ("Concat", typeName);
    }

    const auto convOutput = transformed->get_output_op(1);
    const auto convolution = convOutput->get_input_node_shared_ptr(0);
    const std::string convName = convolution->get_type_name();
    ASSERT_EQ("ConvolutionIE", convName);
}

TEST_P(ConcatWithIntermediateTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
