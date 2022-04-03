// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset9.hpp>

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/eye.hpp"

using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

namespace LayerTestsDefinitions {

std::string EyeLayerTest::getTestCaseName(const testing::TestParamInfo<eyeParams>& obj) {
    std::vector<size_t> inputShape;
    std::vector<int> eyeParams;

    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(inputShape, eyeParams, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << obj.index << "_";
    result << "in_shape=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "prec=" << netPrecision.name() << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

void EyeLayerTest::GenerateInputs() {
    LayerTestsCommon::GenerateInputs();
    // std::cout << "\ngenerate_start\n";
    // size_t it = 0;
    // for (const auto &input : cnnNetwork.getInputsInfo()) {
    //     const auto &info = input.second;
    //     InferenceEngine::Blob::Ptr blob;

    //     blob = make_blob_with_precision(info->getTensorDesc());
    //     blob->allocate();
    //     float* ptr = blob->buffer();

    //     if (it == 0 || it == 1) {
    //         ptr[0] = 3;
    //     } else {
    //         ptr[0] = 0;
    //     }
    //     inputs.push_back(blob);
    //     it++;
    // }
    // std::cout << "\ngenerate_end\n";
}

void EyeLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<int> eyeParams;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, eyeParams, netPrecision, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto rowsPar = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ngraph::Shape{1});
    rowsPar->set_friendly_name("rows");
    auto colsPar = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ngraph::Shape{1});
    colsPar->set_friendly_name("cols");
    auto diagPar = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ngraph::Shape{1});
    diagPar->set_friendly_name("diagInd");
    auto eyelike = std::make_shared<ngraph::opset9::Eye>(rowsPar, colsPar, diagPar, ngraph::element::i32);

    function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(eyelike),
        ngraph::ParameterVector{rowsPar, colsPar, diagPar}, "Eye");

// auto range = std::make_shared<ngraph::opset4::Range>(params[0], params[1], params[2], ngPrc);

//     function = std::make_shared<ngraph::Function>(
//         std::make_shared<ngraph::opset1::Result>(range),
//         params,
//         "Range");

    // auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    // auto params = ngraph::builder::makeParams(ngPrc, {{ 1, 2, 1}});
    // std::vector<int> pooledSpatialShape = {1};
    // ngraph::Shape pooledShape = {pooledSpatialShape.size() };
    // auto pooledParam = ngraph::builder::makeConstant<int32_t>(ngraph::element::i32, pooledShape, pooledSpatialShape);

    // auto adapoolAvg = std::make_shared<ngraph::opset8::AdaptiveAvgPool>(params[0], pooledParam);

    // function = std::make_shared<ngraph::Function>(adapoolAvg->outputs(), params, "AdaPoolAvg");
}
}  // namespace LayerTestsDefinitions
