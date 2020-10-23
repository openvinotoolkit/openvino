// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <ie_plugin_config.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include <legacy/ngraph_ops/power.hpp>
#include "subgraph_tests/softsign.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string SoftsignTest::getTestCaseName(testing::TestParamInfo<softsignParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShape;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration, inputShape) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void SoftsignTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::vector<size_t> inputShape;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { inputShape });

    auto abs = std::make_shared<ngraph::op::Abs>(params[0]);
    auto add = std::make_shared<ngraph::op::PowerIE>(abs, 1, 1, 1);
    auto power = std::make_shared<ngraph::op::PowerIE>(add, -1, 1, 0);
    auto mul = std::make_shared<ngraph::op::Multiply>(power, params[0]);
    ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(mul) };
    function = std::make_shared<ngraph::Function>(results, params, "SoftSignTest");
}

void SoftsignTest::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LoadNetwork();
    Infer();

    const auto& actualOutputs = GetOutputs();
    auto referenceOutputs = CalculateRefs();

    Compare(referenceOutputs, actualOutputs);
}

std::vector<std::vector<std::uint8_t>> SoftsignTest::CalculateRefs() {
    InferenceEngine::Precision netPrecision = std::get<0>(this->GetParam());
    std::vector<size_t> inputShape = std::get<3>(this->GetParam());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { inputShape });
    auto abs = std::make_shared<ngraph::op::Abs>(params[0]);
    auto power = std::make_shared<ngraph::op::PowerIE>(abs, -1, 1, 1);
    auto mul = std::make_shared<ngraph::op::Multiply>(power, params[0]);
    ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(mul) };
    auto reference_model = std::make_shared<ngraph::Function>(results, params, "SoftSignTest");

    auto refCnnNetwork = InferenceEngine::CNNNetwork{ reference_model };
    auto refExecutableNetwork = core->LoadNetwork(refCnnNetwork, targetDevice);

    auto refInferRequest = refExecutableNetwork.CreateInferRequest();
    std::vector<InferenceEngine::InputInfo::Ptr> refInfos;
    for (const auto& input : refCnnNetwork.getInputsInfo()) {
        const auto& info = input.second;
        refInfos.push_back(info);
    }

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        const auto& info = refInfos[i];

        refInferRequest.SetBlob(info->name(), input);
    }

    refInferRequest.Infer();

    auto refOutputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto& output : refCnnNetwork.getOutputsInfo()) {
        const auto& name = output.first;
        refOutputs.push_back(refInferRequest.GetBlob(name));
    }

    auto referenceOutputs = std::vector<std::vector<std::uint8_t>>(refOutputs.size());
    for (std::size_t i = 0; i < refOutputs.size(); ++i) {
        const auto& reference = refOutputs[i];
        const auto refSize = reference->byteSize();

        auto& expectedOutput = referenceOutputs[i];
        expectedOutput.resize(refSize);

        auto refMemory = InferenceEngine::as<InferenceEngine::MemoryBlob>(reference);
        IE_ASSERT(refMemory);
        const auto refLockedMemory = refMemory->wmap();
        const auto referenceBuffer = refLockedMemory.as<const std::uint8_t*>();

        std::copy(referenceBuffer, referenceBuffer + refSize, expectedOutput.data());
    }

    return referenceOutputs;
}

TEST_P(SoftsignTest, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
