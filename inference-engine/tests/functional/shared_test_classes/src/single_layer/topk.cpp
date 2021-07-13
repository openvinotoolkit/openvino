// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/topk.hpp"

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

    // Spec TopK_3.md allows to use unstable sorting, thus
    // a. Skip comparing of index results, because an element in actual index tensor can be different with
    //    its counterpart in expected index tensor
    // b. If SortType is SORT_INDICES or NONE, the test program still needs to apply std::sort for all pairs
    //    of 1xk value vectors in expected and actual output tensor before comparing them
    size_t axis_idx = axis < 0 ? static_cast<size_t>(axis + static_cast<int64_t>(inputShape.size())) : static_cast<size_t>(axis);
    if (sort == ngraph::opset4::TopK::SortType::SORT_VALUES)
        setCustomizedCompare(false, true, static_cast<size_t>(keepK), axis_idx, inputShape);
    else
        setCustomizedCompare(false, false, static_cast<size_t>(keepK), axis_idx, inputShape);

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
}  // namespace LayerTestsDefinitions
