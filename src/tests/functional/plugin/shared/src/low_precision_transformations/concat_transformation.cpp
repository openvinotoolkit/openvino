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

void ConcatTransformation::SetUp() {
    abs_threshold = 0.1;
    rel_threshold = 4.2;

    ngraph::PartialShape inputShape;
    ngraph::element::Type precision;
    ConcatTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    std::vector<ngraph::PartialShape> inputs;
    if (testValues.input_constant1 == nullptr) {
        inputs.push_back(inputShape);
    }
    if (testValues.input_constant2 == nullptr) {
        inputs.push_back(inputShape);
    }
    init_input_shapes(inputs);

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
    run();
};

}  // namespace LayerTestsDefinitions
