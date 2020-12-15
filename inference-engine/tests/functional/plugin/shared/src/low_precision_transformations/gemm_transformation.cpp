// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/gemm_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string GemmTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params);
}

void GemmTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const float low = params.precisionsOnActivations[0] == ngraph::element::u8 ? 0.f : -128.f;
    const float high = params.precisionsOnActivations[0] == ngraph::element::u8 ? 255.f : 127.f;

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(
        input1, ngPrecision, 256ul, { 1ul },
        { low / 4.f }, { high / 4.f }, { low / 4.f }, { high / 4.f });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(
        input2, ngPrecision, 256ul, { 1ul },
        { low / 8.f }, { high / 8.f }, { low / 8.f }, { high / 8.f });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const auto matMul = std::make_shared<ngraph::opset1::MatMul>(
        fakeQuantize1->output(0),
        fakeQuantize2->output(0),
        false,
        false);
    matMul->set_friendly_name("matMul");

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(matMul)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { input1, input2 }, "GemmTransformation");
}

TEST_P(GemmTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
