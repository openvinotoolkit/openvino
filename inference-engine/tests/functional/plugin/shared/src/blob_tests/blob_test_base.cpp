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
    std::tuple<networkPreprocessFuncType, generateInputFuncType, generateReferenceFuncType, makeTestBlobFuncType, teardownFuncType> fnSet;
    size_t inferCount;
    size_t batchSize;
    bool asyncExec;
    std::string targetDevice;
    std::map<std::string, std::string> config;

    std::tie(precision, fn_pair, fnSet, inferCount, batchSize, asyncExec, targetDevice, config) = obj.param;
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
    std::tuple<networkPreprocessFuncType, generateInputFuncType, generateReferenceFuncType, makeTestBlobFuncType, teardownFuncType> fnSet;
    std::tie(precision, fn_pair, fnSet, inferCount, batchSize, asyncExecution, targetDevice, config) = this->GetParam();
    std::tie(preprocessFn, generateInputFn, generateReferenceFn, makeBlobFn, teardownFn) = fnSet;

    configuration.insert(config.begin(), config.end());

    auto input_sizes = fn_pair.second;
    input_sizes[0] = batchSize;

    function = fn_pair.first(input_sizes, precision);
}

InferenceEngine::Blob::Ptr BlobTestBase::GenerateInput(const InferenceEngine::InputInfo& info) const {
    if (generateInputFn) return generateInputFn(info);

    static int seed = 0;
    seed++;
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 5, -2, 100, seed);
}

std::vector<std::vector<uint8_t>> BlobTestBase::CalculateReference() {
    if (generateReferenceFn) {
        return generateReferenceFn(cnnNetwork, inputs);
    } else {
        return CalculateRefs();
    }
}

InferenceEngine::Blob::Ptr BlobTestBase::PrepareTestBlob(InferenceEngine::Blob::Ptr inputBlob) {
    if (makeBlobFn) {
        return makeBlobFn(inputBlob, executableNetwork);
    } else {
        return inputBlob;
    }
}

void BlobTestBase::Infer() {
    std::vector<InferenceEngine::InferRequest> inferRequests;
    for (int i = 0; i < inferCount; i++) {
        // Clean after old inference and create a new one
        inputs.clear();
        inferRequests.push_back(executableNetwork.CreateInferRequest());
        inferRequest = inferRequests.back();

        // Generrate new set of inputs
        GenerateInputs();
        refInputBlobs.push_back(inputs);

        // Calculate reference output for this inference
        referenceOutputs.push_back(CalculateReference());

        // Prepare new set of blobs based on original input blobs and set them for the inference
        inputs.clear();
        for (const auto &input : executableNetwork.GetInputsInfo()) {
            const auto &info = input.second;
            auto testBlob =  PrepareTestBlob(refInputBlobs[i][inputs.size()]);
            inferRequests.back().SetBlob(info->name(), testBlob);
            inputs.push_back(testBlob);
        }
    }

    // execute inrefence
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

    // Get outputs out of inferences
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
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    ConfigureNetwork();
    if (preprocessFn) preprocessFn(cnnNetwork);
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
    GenerateInputs();
    Infer();
    Validate();

    if (teardownFn) teardownFn();
}

TEST_P(BlobTestBase, CompareWithRefs) {
    Run();
};
} // namespace BlobTestsDefinitions
