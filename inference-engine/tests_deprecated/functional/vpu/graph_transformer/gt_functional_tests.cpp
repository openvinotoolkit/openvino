// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gt_functional_tests.hpp"

#include <vpu/utils/logger.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/graph_transformer_internal.hpp>

using namespace InferenceEngine;
using namespace vpu;

namespace  {

}  // namespace

void graphTransformerFunctionalTests::SetUp() {
    vpuLayersTests::SetUp();

    _stageBuilder = std::make_shared<StageBuilder>();
    _platform = CheckMyriadX() ? Platform::MYRIAD_X : Platform::MYRIAD_2;
}

void graphTransformerFunctionalTests::CreateModel() {
    const auto compilerLog = std::make_shared<Logger>("Test", LogLevel::Info, consoleOutput());
    CompileEnv::init(_platform, _compilationConfig, compilerLog);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });
    const auto& env = CompileEnv::get();

    auto unitTest = testing::UnitTest::GetInstance();
    IE_ASSERT(unitTest != nullptr);
    auto curTestInfo = unitTest->current_test_info();
    IE_ASSERT(curTestInfo != nullptr);

    _gtModel = std::make_shared<ModelObj>(
                formatString("%s/%s", curTestInfo->test_case_name(), curTestInfo->name()));
    _gtModel->attrs().set<Resources>("resources", env.resources);
    _gtModel->attrs().set<int>("index", 1);
}

void graphTransformerFunctionalTests::PrepareGraphCompilation() {
    SetSeed(DEFAULT_SEED_VALUE);
    _compilationConfig = CompilationConfig();
    _inputsInfo.clear();
    _outputsInfo.clear();
    _inputMap.clear();
    _outputMap.clear();

    // Executable network holds its device in booted & busy state.
    // For the new network plugin tries to find new free device first (already booted or not booted),
    // then to reuse busy devices. If we release the executable network, it marks its device as free and booted.
    // Next network will find such device and will use it without boot, which is the fastest case.
    _executableNetwork = {};
    _inferRequest = {};

    CreateModel();
}

void graphTransformerFunctionalTests::InitializeInputData(const DataDesc& inputDataDesc) {
    auto input = _gtModel->addInputData("Input", inputDataDesc);
    _gtModel->attrs().set<int>("numInputs", 1);

    InputInfo::Ptr inputInfoPtr(new InputInfo());
    inputInfoPtr->setInputData(std::make_shared<InferenceEngine::Data>("Input", inputDataDesc.toTensorDesc()));
    _inputsInfo["Input"] = inputInfoPtr;

    _dataIntermediate  = input;
}

vpu::Data graphTransformerFunctionalTests::InitializeOutputData(const DataDesc& outputDataDesc) {
    vpu::Data output = _gtModel->addOutputData("Output", outputDataDesc);
    _gtModel->attrs().set<int>("numOutputs", 1);

    _outputsInfo["Output"] = std::make_shared<InferenceEngine::Data>("Output", outputDataDesc.toTensorDesc());
    return output;
}

int64_t graphTransformerFunctionalTests::CompileAndInfer(Blob::Ptr& inputBlob, Blob::Ptr& outputBlob, bool lockLayout) {
    const auto compilerLog = std::make_shared<Logger>(
                "Test",
                LogLevel::Info,
                consoleOutput());

    auto compiledGraph = compileModel(
                _gtModel,
                _platform,
                _compilationConfig,
                compilerLog);

    std::istringstream instream(std::string(compiledGraph->blob.data(), compiledGraph->blob.size()));

    _executableNetwork = _vpuPluginPtr->ImportNetwork(instream, _config);
    auto inferRequest = _executableNetwork.CreateInferRequest();
    _inferRequest = inferRequest;

    genInputBlobs(lockLayout);
    genOutputBlobs(lockLayout);

    IE_ASSERT(Infer());

    auto perfMap = _inferRequest.GetPerformanceCounts();

    int64_t executionMicroseconds = 0;
    for (const auto& perfPair : perfMap) {
        const InferenceEngine::InferenceEngineProfileInfo& info = perfPair.second;
        if (info.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
            executionMicroseconds += info.realTime_uSec;
        }
    }
    inputBlob = _inputMap.begin()->second;
    outputBlob = _outputMap.begin()->second;
    return executionMicroseconds;
}
