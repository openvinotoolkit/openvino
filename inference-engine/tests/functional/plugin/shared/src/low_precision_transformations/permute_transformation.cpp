// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/permute_transformation.hpp"

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
#include "ngraph_functions/builders.hpp"


namespace LayerTestsDefinitions {

std::string PermuteTransformation::getTestCaseName(testing::TestParamInfo<PermuteTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool perTensor;
    bool transposeChannelDim;
    std::tie(netPrecision, inputShapes, targetDevice, params, perTensor, transposeChannelDim) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << targetDevice << "_" << toString(params) <<
        (perTensor ? "_perTensor" : "_perChannel") <<
        (transposeChannelDim ? "_transposeChannelDim" : "_notTransposeChannelDim");
    return result.str();
}

void PermuteTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool perTensor;
    bool transposeChannelDim;
    std::tie(netPrecision, inputShape, targetDevice, params, perTensor, transposeChannelDim) = this->GetParam();

    const auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");

    const float k = 50.f;
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(input1, precision, 256ul, { 1ul }, { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
    input2->set_friendly_name("fakeQuantize1");
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(input2, precision, 256ul, { 1ul }, { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
    input2->set_friendly_name("fakeQuantize2");
    const auto matMul = std::make_shared<ngraph::opset1::MatMul>(fakeQuantize1, fakeQuantize2, false, false);
    input2->set_friendly_name("matMul");
    const auto transpose = std::make_shared<ngraph::opset1::Transpose>(
        matMul,
        ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4ul }, { 0, 2, 1, 3 }));
    transpose->set_friendly_name("transpose");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(transpose) };
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input1, input2 }, "PermuteTransformation");
}

TEST_P(PermuteTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
