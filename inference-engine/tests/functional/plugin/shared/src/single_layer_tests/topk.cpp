// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/topk.hpp"

namespace LayerTestsDefinitions {
    std::string TopKLayerTest::getTestCaseName(testing::TestParamInfo<TopKParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    int64_t keepK, axis;
    ngraph::opset4::TopK::Mode mode;
    ngraph::opset4::TopK::SortType sort;
    std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "k=" << keepK << "_";
    result << "axis=" << axis << "_";
    result << "mode=" << mode << "_";
    result << "sort=" << sort << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void TopKLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    int64_t keepK, axis;
    ngraph::opset4::TopK::Mode mode;
    ngraph::opset4::TopK::SortType sort;
    std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramIn = ngraph::helpers::convert2OutputVector(
                        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto k = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, &keepK);
    auto topk = std::dynamic_pointer_cast<ngraph::opset4::TopK>(
            std::make_shared<ngraph::opset4::TopK>(paramIn[0], k, axis, mode, sort));

    ngraph::ResultVector results;
    for (int i = 0; i < topk->get_output_size(); i++) {
        results.push_back(std::make_shared<ngraph::opset4::Result>(topk->output(i)));
    }
    function = std::make_shared<ngraph::Function>(results, params, "TopK");
}

TEST_P(TopKLayerTest, CompareWithRefsDynamicBath) {
    Run();
}
}  // namespace LayerTestsDefinitions