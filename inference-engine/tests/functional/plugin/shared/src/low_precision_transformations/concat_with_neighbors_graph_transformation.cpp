// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_neighbors_graph_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/builders.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"

namespace LayerTestsDefinitions {

std::string ConcatWithNeighborsGraphTransformation::getTestCaseName(testing::TestParamInfo<ConcatNeighboringGraphTransformationParams> obj) {
    ngraph::element::Type precision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, inputShapes, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(precision, inputShapes, targetDevice, params);
}

InferenceEngine::Blob::Ptr ConcatWithNeighborsGraphTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    if ((info.name() != "input1") && (info.name() != "input2") && (info.name() != "input3")) {
        IE_THROW() << "unexpected input name " << info.name();
    }
    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(params.precisionsOnActivations[0], info.getTensorDesc(), k);
}

void ConcatWithNeighborsGraphTransformation::SetUp() {
    threshold = 2.e-2;
    ngraph::element::Type ngPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(ngPrecision, inputShape, targetDevice, params) = this->GetParam();

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithNeighbors(
        ngPrecision,
        inputShape,
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} },
        "concat",
        "");

    validate();
}

void ConcatWithNeighborsGraphTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));
    ASSERT_EQ(2ul, transformed->get_output_size());

    for (size_t i = 0; i < 2ul; ++i) {
        const auto concatOutput = transformed->get_output_op(0);
        const auto scaleShift = concatOutput->get_input_node_shared_ptr(0);
        const std::string typeName = scaleShift->get_type_name();
        ASSERT_EQ("ScaleShiftIE", typeName);
    }
}

TEST_P(ConcatWithNeighborsGraphTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
