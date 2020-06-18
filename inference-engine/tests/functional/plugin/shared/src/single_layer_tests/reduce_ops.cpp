// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/reduce_ops.hpp"

namespace LayerTestsDefinitions {

std::string ReduceOpsLayerTest::getTestCaseName(testing::TestParamInfo<reduceMeanParams> obj) {
    InferenceEngine::Precision netPrecision;
    bool keepDims;
    ngraph::helpers::ReductionType reductionType;
    std::vector<size_t> inputShape;
    std::vector<int> axes;
    std::string targetDevice;
    std::tie(axes, keepDims, reductionType, netPrecision, inputShape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "axes=" << CommonTestUtils::vec2str(axes) << "_";
    result << "type=" << reductionType << "_";
    if (keepDims) result << "KeepDims_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ReduceOpsLayerTest::SetUp() {
    // TODO: Issue 33151
    // Failed to create function on SetUp stage with some parameters
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Precision netPrecision;
    bool keepDims;
    ngraph::helpers::ReductionType reductionType;
    std::vector<size_t> inputShape;
    std::vector<int> axes;
    std::tie(axes, keepDims, reductionType, netPrecision, inputShape, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    const auto reduce = ngraph::builder::makeReduce(paramOuts, axes, keepDims, reductionType);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(reduce)};
    function = std::make_shared<ngraph::Function>(results, params, "Reduce");
}

TEST_P(ReduceOpsLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions