// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_with_incorrect_weights.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/convolution.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionWIthIncorrectWeightsTransformation::getTestCaseName(const testing::TestParamInfo<ConvolutionWIthIncorrectWeightsParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ConvolutionWIthIncorrectWeightsParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) <<
           (param.isCorrect ? "_correct_weights" : "_incorrect_weights") <<
        (param.fakeQuantizeOnData.empty() ? "_noFqOnActivations" : "") <<
        (param.fakeQuantizeOnWeights.empty() ? "_noFqOnWeights" : "");
    return result.str();
}

void ConvolutionWIthIncorrectWeightsTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ConvolutionWIthIncorrectWeightsParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::ConvolutionFunction::getOriginalWithIncorrectWeights(
        inputShape,
        netPrecision,
        param.fakeQuantizeOnWeights,
        param.fakeQuantizeOnData,
        param.isCorrect);
}

TEST_P(ConvolutionWIthIncorrectWeightsTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
