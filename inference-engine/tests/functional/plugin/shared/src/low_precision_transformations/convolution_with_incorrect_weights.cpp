// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_with_incorrect_weights.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/low_precision_transformations/convolution_function.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionWIthIncorrectWeightsTransformation::getTestCaseName(testing::TestParamInfo<ConvolutionWIthIncorrectWeightsParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ConvolutionWIthIncorrectWeightsParam param;
    std::tie(netPrecision, inputShapes, targetDevice, params, version, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version) <<
        (param.isCorrect ? "_correct_weights" : "_incorrect_weights") <<
        (param.fakeQuantizeOnData.empty() ? "_noFqOnActivations" : "") <<
        (param.fakeQuantizeOnWeights.empty() ? "_noFqOnWeights" : "");
    return result.str();
}

void ConvolutionWIthIncorrectWeightsTransformation::SetUp() {
    threshold = 0.1f;

    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ConvolutionWIthIncorrectWeightsParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, version, param) = this->GetParam();
    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::ConvolutionFunction::getOriginalWithIncorrectWeights(
        inputShape,
        precision,
        param.fakeQuantizeOnWeights,
        param.fakeQuantizeOnData,
        param.isCorrect);
}
TEST_P(ConvolutionWIthIncorrectWeightsTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
