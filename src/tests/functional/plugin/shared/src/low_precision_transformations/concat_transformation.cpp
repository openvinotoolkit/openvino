// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatTransformation::getTestCaseName(const testing::TestParamInfo<ConcatTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ConcatTransformationTestValues testValues;
    std::tie(precision, inputShapes, targetDevice, testValues) = obj.param;

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, inputShapes, targetDevice, params) <<
        testValues.fqOnData1 <<
        testValues.dequantization1 <<
        testValues.fqOnData2 <<
        testValues.dequantization2;
    return result.str();
}

InferenceEngine::Blob::Ptr ConcatTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::PartialShape inputShape;
    ngraph::element::Type netPrecision;
    std::string targetDevice;
    ConcatTransformationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, testValues) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(ngraph::element::u8, info.getTensorDesc(), k);
}

void ConcatTransformation::SetUp() {
    ngraph::PartialShape inputShape;
    ngraph::element::Type precision;
    ConcatTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::ConcatFunction::getOriginal(
        precision,
        inputShape,
        testValues.input_constant1,
        testValues.fqOnData1,
        testValues.dequantization1,
        testValues.input_constant2,
        testValues.fqOnData2,
        testValues.dequantization2);
}

TEST_P(ConcatTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
