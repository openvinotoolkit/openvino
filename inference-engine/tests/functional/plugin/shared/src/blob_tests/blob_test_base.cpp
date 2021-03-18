// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include "blob_tests/blob_test_base.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace BlobTestsDefinitions {

std::string BlobTestBase::getTestCaseName(testing::TestParamInfo<BlobTestBaseParams> obj) {
    ngraph::element::Type precision;
    std::pair<nghraphSubgraphFuncType, std::vector<size_t>> fn_pair;
    makeTestBlobFuncType makeBlobFn;
    std::function<void()> teardownFn;
    size_t inferCount;
    size_t batchSize;
    bool asyncExec;
    std::string targetDevice;
    std::map<std::string, std::string> config;

    std::tie(precision, fn_pair, makeBlobFn, teardownFn, inferCount, batchSize, asyncExec, targetDevice, config) = obj.param;
    auto function = fn_pair.first(fn_pair.second, precision);
    std::ostringstream result;
    result <<
        "NGPrc=" << precision << "_" <<
        "function=" << function->get_friendly_name() << "_" <<
        "InferCount=" << inferCount << "_" <<
        "BS=" << batchSize << "_" <<
        "async=" << asyncExec << "_" <<
        "device=" << targetDevice;

    return result.str();
}

void BlobTestBase::SetUp() {
    ngraph::element::Type precision;
    std::pair<nghraphSubgraphFuncType, std::vector<size_t>> fn_pair;
    size_t batchSize;
    std::map<std::string, std::string> config;

    std::tie(precision, fn_pair, makeBlobFn, teardownFn, inferCount, batchSize, asyncExecution, targetDevice, config) = this->GetParam();

    configuration.insert(config.begin(), config.end());

    auto input_sizes = fn_pair.second;
    input_sizes[0] = batchSize;

    function = fn_pair.first(input_sizes, precision);
}

void BlobTestBase::Infer() {
    std::vector<InferenceEngine::InferRequest> inferRequests;
    for (int i = 0; i < inferCount; i++) {
        inferRequests.push_back(executableNetwork.CreateInferRequest());
        inferRequest = inferRequests.back();
        inputs.clear();
        auto inputsInfo = executableNetwork.GetInputsInfo();
        for (auto& input : inputsInfo) {
            const auto &info = input.second;
            auto refInput = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), 5, -2, 100, i); 
            inputs.push_back(refInput);
        }
        refInputBlobs.push_back(inputs);
        referenceOutputs.push_back(CalculateRefs());

        inputs.clear();
        for (const auto &input : executableNetwork.GetInputsInfo()) {
            const auto &info = input.second;
            auto testBlob =  makeBlobFn(refInputBlobs[i][inputs.size()], executableNetwork);
            inferRequests.back().SetBlob(info->name(), testBlob);
            inputs.push_back(testBlob);
        }
    }

    for (auto& inferRequest : inferRequests) {
        if (asyncExecution) {
            inferRequest.StartAsync();
        } else {
            inferRequest.Infer();
        }
    }

    if (asyncExecution) {
        for (auto& inferRequest : inferRequests) {
            auto status = inferRequest.Wait(10000);
            if (status != InferenceEngine::StatusCode::OK) {
                GTEST_FAIL() << "Inference request status after wait is not OK";
            }
        }
    }

    for (int i = 0; i < inferRequests.size(); i++) {
        auto outputs = std::vector<InferenceEngine::Blob::Ptr>{};
        for (const auto &output : executableNetwork.GetOutputsInfo()) {
            const auto &name = output.first;
            outputs.push_back(inferRequests[i].GetBlob(name));
        }
        actualOutputs.push_back(outputs);
    }
}

void BlobTestBase::Validate() {
    this->threshold = 0.1f;
    for (int i = 0; i < inferCount; i++) {
        Compare(referenceOutputs[i], actualOutputs[i]);
    }
}

void BlobTestBase::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    LoadNetwork();
    GenerateInputs();
    Infer();
    Validate();
    teardownFn();
}

TEST_P(BlobTestBase, CompareWithRefs) {
    Run();
};
}
