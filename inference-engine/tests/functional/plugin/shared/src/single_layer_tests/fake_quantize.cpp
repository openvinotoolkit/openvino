// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/fake_quantize.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeLayerTest::getTestCaseName(testing::TestParamInfo<fqLayerTestParamsSet> obj) {
    fqSpecificParams fqParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::pair<std::string, std::map<std::string, std::string>> config;
    std::tie(fqParams, netPrecision, inputShapes, targetDevice, config) = obj.param;
    size_t levels;
    std::vector<size_t> constShape;
    std::vector<float> fqDirectArgs;
    std::tie(levels, constShape, fqDirectArgs) = fqParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "CS=" << CommonTestUtils::vec2str(constShape) << "_";
    result << "LEVELS=" << levels << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    if (!config.first.empty()) {
        result << "_targetConfig=" << config.first;
    }
    if (!fqDirectArgs.empty()) {
        result << "_fqArgs=" << fqDirectArgs[0] << "_" << fqDirectArgs[1] << "_" << fqDirectArgs[2] << "_" << fqDirectArgs[3];
    }
    return result.str();
}

void FakeQuantizeLayerTest::SetUp() {
    fqSpecificParams fqParams;
    std::vector<size_t> inputShape;
    std::pair<std::string, std::map<std::string, std::string>> config;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(fqParams, netPrecision, inputShape, targetDevice, config) = this->GetParam();
    InferenceEngine::SizeVector kernel, stride, dilation;
    size_t levels;
    std::vector<size_t> constShape;
    std::vector<float> fqDirectArg;
    std::tie(levels, constShape, fqDirectArg) = fqParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::shared_ptr<ngraph::Node> fakeQNode;
    if (fqDirectArg.empty()) {
        fakeQNode = ngraph::builder::makeFakeQuantize(paramOuts[0], ngPrc, levels, constShape);
    } else {
        fakeQNode = ngraph::builder::makeFakeQuantize(
            paramOuts[0],
            ngPrc,
            levels,
            constShape,
            {fqDirectArg[0]},
            {fqDirectArg[1]},
            {fqDirectArg[2]},
            {fqDirectArg[3]});
    }


    auto fq = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(fakeQNode);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(fq)};
    function = std::make_shared<ngraph::Function>(results, params, "fakeQuantize");

    configuration = config.second;
}

TEST_P(FakeQuantizeLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
