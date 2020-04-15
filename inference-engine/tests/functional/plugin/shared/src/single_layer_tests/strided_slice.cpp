// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ie_core.hpp>
#include <ngraph_functions/builders.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "single_layer_tests/strided_slice.hpp"

namespace LayerTestsDefinitions {

std::string StridedSliceLayerTest::getTestCaseName(const testing::TestParamInfo<stridedSliceParamsTuple> &obj) {
    InferenceEngine::SizeVector inputShape;
    std::vector<int64_t> begin, end, stride;
    std::vector<int64_t> begin_mask, new_axis_mask, end_mask, shrink_mask, ellipsis_mask;
    InferenceEngine::Precision inPrc, netPrc;
    std::string targetName;
    std::tie(inputShape, begin, end, stride, begin_mask, end_mask, new_axis_mask, shrink_mask, ellipsis_mask, inPrc, netPrc, targetName) = obj.param;
    std::ostringstream result;
    result << "inShape=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "begin=" << CommonTestUtils::vec2str(begin) << "_";
    result << "end=" << CommonTestUtils::vec2str(end) << "_";
    result << "stride=" << CommonTestUtils::vec2str(stride) << "_";
    result << "begin_m=" << CommonTestUtils::vec2str(begin_mask) << "_";
    result << "end_m=" << CommonTestUtils::vec2str(end_mask) << "_";
    result << "new_axis_m=" << CommonTestUtils::vec2str(new_axis_mask) << "_";
    result << "shrink_m=" << CommonTestUtils::vec2str(shrink_mask) << "_";
    result << "ellipsis_m=" << CommonTestUtils::vec2str(ellipsis_mask) << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void StridedSliceLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    std::vector<int64_t> begin, end, stride;
    std::vector<int64_t> begin_mask, end_mask, new_axis_mask, shrink_mask, ellipsis_mask;
    std::tie(inputShape, begin, end, stride, begin_mask, end_mask, new_axis_mask, shrink_mask, ellipsis_mask,
             inputPrecision, netPrecision, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto ss = ngraph::builder::makeStridedSlice(paramOuts[0], begin, end, stride, ngPrc, begin_mask, end_mask, new_axis_mask, shrink_mask, ellipsis_mask);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ss)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "StridedSlice");
}

TEST_P(StridedSliceLayerTest, CompareWithRefs) {
    inferAndValidate();
}

}  // namespace LayerTestsDefinitions
