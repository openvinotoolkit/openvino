// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gt_functional_tests.hpp"

#include <vpu/utils/logger.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/graph_transformer_internal.hpp>
#include <vpu/configuration/options/log_level.hpp>
#include <vpu/configuration/options/copy_optimization.hpp>
#include <vpu/configuration/options/protocol.hpp>
#include <vpu/configuration/options/power_config.hpp>
#include <vpu/configuration/options/hw_acceleration.hpp>
#include <vpu/configuration/options/hw_extra_split.hpp>
#include <vpu/configuration/options/hw_pool_conv_merge.hpp>
#include <vpu/configuration/options/hw_black_list.hpp>
#include <vpu/configuration/options/hw_inject_stages.hpp>
#include <vpu/configuration/options/hw_dilation.hpp>
#include <vpu/configuration/options/tiling_cmx_limit_kb.hpp>
#include <vpu/configuration/options/watchdog_interval.hpp>
#include <vpu/configuration/options/enable_receiving_tensor_time.hpp>
#include <vpu/configuration/options/perf_report_mode.hpp>
#include <vpu/configuration/options/perf_count.hpp>
#include <vpu/configuration/options/pack_data_in_cmx.hpp>
#include <vpu/configuration/options/number_of_shaves.hpp>
#include <vpu/configuration/options/number_of_cmx_slices.hpp>
#include <vpu/configuration/options/throughput_streams.hpp>
#include <vpu/configuration/options/ir_with_scales_directory.hpp>
#include <vpu/configuration/options/tensor_strides.hpp>
#include <vpu/configuration/options/ignore_unknown_layers.hpp>
#include <vpu/configuration/options/force_pure_tensor_iterator.hpp>
#include <vpu/configuration/options/exclusive_async_requests.hpp>
#include <vpu/configuration/options/enable_weights_analysis.hpp>

using namespace InferenceEngine;
using namespace vpu;

namespace  {

}  // namespace

void graphTransformerFunctionalTests::SetUp() {
    vpuLayersTests::SetUp();

    _stageBuilder = std::make_shared<StageBuilder>();
    _platform = CheckMyriadX() ? ncDevicePlatform_t::NC_MYRIAD_X : ncDevicePlatform_t::NC_MYRIAD_2;
}

void graphTransformerFunctionalTests::CreateModel() {
    const auto compilerLog = std::make_shared<Logger>("Test", LogLevel::Info, consoleOutput());
    CompileEnv::init(_platform, _configuration, compilerLog);
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

    _configuration.registerOption<LogLevelOption>();
    _configuration.registerOption<CopyOptimizationOption>();
    _configuration.registerOption<ProtocolOption>();
    _configuration.registerOption<PowerConfigOption>();
    _configuration.registerOption<HwAccelerationOption>();
    _configuration.registerOption<HwExtraSplitOption>();
    _configuration.registerOption<HwPoolConvMergeOption>();
    _configuration.registerOption<HwBlackListOption>();
    _configuration.registerOption<HwInjectStagesOption>();
    _configuration.registerOption<HwDilationOption>();
    _configuration.registerOption<TilingCMXLimitKBOption>();
    _configuration.registerOption<WatchdogIntervalOption>();
    _configuration.registerOption<EnableReceivingTensorTimeOption>();
    _configuration.registerOption<PerfReportModeOption>();
    _configuration.registerOption<PerfCountOption>();
    _configuration.registerOption<PackDataInCMXOption>();
    _configuration.registerOption<NumberOfSHAVEsOption>();
    _configuration.registerOption<NumberOfCMXSlicesOption>();
    _configuration.registerOption<ThroughputStreamsOption>();
    _configuration.registerOption<IRWithScalesDirectoryOption>();
    _configuration.registerOption<TensorStridesOption>();
    _configuration.registerOption<IgnoreUnknownLayersOption>();
    _configuration.registerOption<ForcePureTensorIteratorOption>();
    _configuration.registerOption<ExclusiveAsyncRequestsOption>();
    _configuration.registerOption<EnableWeightsAnalysisOption>();

IE_SUPPRESS_DEPRECATED_START
    _configuration.registerDeprecatedOption<LogLevelOption>(VPU_CONFIG_KEY(LOG_LEVEL));
    _configuration.registerDeprecatedOption<ProtocolOption>(VPU_MYRIAD_CONFIG_KEY(PROTOCOL));
    _configuration.registerDeprecatedOption<HwAccelerationOption>(VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION));
    _configuration.registerDeprecatedOption<EnableReceivingTensorTimeOption>(VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME));
IE_SUPPRESS_DEPRECATED_END

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
                _configuration,
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
