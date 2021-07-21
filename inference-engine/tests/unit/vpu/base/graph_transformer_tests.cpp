// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include <vpu/utils/io.hpp>
#include <vpu/private_plugin_config.hpp>

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
#include <vpu/configuration/options/enable_tensor_iterator_unrolling.hpp>
#include <vpu/configuration/options/exclusive_async_requests.hpp>
#include <vpu/configuration/options/enable_weights_analysis.hpp>
#include <vpu/configuration/options/enable_repl_with_screlu.hpp>
#include <vpu/configuration/options/enable_permute_merging.hpp>
#include <vpu/configuration/options/enable_memory_types_annotation.hpp>
#include <vpu/configuration/options/dump_internal_graph_file_name.hpp>
#include <vpu/configuration/options/dump_all_passes_directory.hpp>
#include <vpu/configuration/options/dump_all_passes.hpp>
#include <vpu/configuration/options/disable_convert_stages.hpp>
#include <vpu/configuration/options/disable_reorder.hpp>
#include <vpu/configuration/options/device_id.hpp>
#include <vpu/configuration/options/device_connect_timeout.hpp>
#include <vpu/configuration/options/detect_network_batch.hpp>
#include <vpu/configuration/options/custom_layers.hpp>
#include <vpu/configuration/options/config_file.hpp>
#include <vpu/configuration/options/memory_type.hpp>
#include <vpu/configuration/options/enable_force_reset.hpp>
#include <vpu/configuration/options/platform.hpp>
#include <vpu/configuration/options/check_preprocessing_inside_model.hpp>
#include <vpu/configuration/options/enable_early_eltwise_relu_fusion.hpp>
#include <vpu/configuration/options/enable_custom_reshape_param.hpp>
#include <vpu/configuration/options/none_layers.hpp>
#include <vpu/configuration/options/enable_async_dma.hpp>

#include <atomic>
#include <iomanip>

namespace vpu {

StagePtr TestStage::cloneImpl() const {
    return std::make_shared<TestStage>(*this);
}

namespace {

template <typename Value>
void setInOutPortInfo(
        const Stage& stage,
        const std::string& attrBaseName,
        StageDataInfo<Value>& info) {
    auto inAttrName = formatString("test_input_%s_info", attrBaseName);
    auto outAttrName = formatString("test_output_%s_info", attrBaseName);

    if (stage->attrs().has(inAttrName)) {
        const auto& inputInfo = stage->attrs().get<InOutPortMap<Value>>(inAttrName);

        for (const auto& p : inputInfo) {
            info.setInput(stage->inputEdge(p.first), p.second);
        }
    }

    if (stage->attrs().has(outAttrName)) {
        const auto& outputInfo = stage->attrs().get<InOutPortMap<Value>>(outAttrName);

        for (const auto& p : outputInfo) {
            info.setOutput(stage->outputEdge(p.first), p.second);
        }
    }
}

} // namespace

InputInfo InputInfo::fromNetwork(int ind) {
    InputInfo info;
    info.type = InputType::Original;
    info.originalInputInd = ind;
    return info;
}

InputInfo InputInfo::fromPrevStage(int ind, int outputInd) {
    InputInfo info;
    info.type = InputType::PrevStageOutput;
    info.prevStageInd = ind;
    info.prevStageOutputInd = outputInd;
    return info;
}

InputInfo& InputInfo::output(int ind) {
    assert(type == InputType::PrevStageOutput);
    prevStageOutputInd = ind;
    return *this;
}

OutputInfo OutputInfo::fromNetwork(int ind) {
    OutputInfo info;
    info.type = OutputType::Original;
    info.originalOutputInd = ind;
    return info;
}

InputInfo InputInfo::constant(const DataDesc& desc) {
    InputInfo info;
    info.type = InputType::Constant;
    info.desc = desc;
    return info;
}

OutputInfo OutputInfo::intermediate(const DataDesc& desc) {
    OutputInfo info;
    info.type = OutputType::Intermediate;
    info.desc = desc;
    return info;
}

OutputInfo OutputInfo::intermediate(MemoryType memReq) {
    OutputInfo info;
    info.type = OutputType::Intermediate;
    info.desc = DataDesc{};
    info.memReq = memReq;
    return info;
}

void TestStage::propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) {
    setInOutPortInfo(this, "DataOrder", orderInfo);
}

void TestStage::getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) {
    setInOutPortInfo(this, "Strides", stridesInfo);
}

void TestStage::finalizeDataLayoutImpl() {
}

void TestStage::getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) {
    setInOutPortInfo(this, "Batch", batchInfo);

    if (attrs().has("test_input_Batch_info")) {
        for (const auto& outEdge : outputEdges()) {
            batchInfo.setOutput(outEdge, BatchSupport::Split);
        }
    }
}

void TestStage::serializeParamsImpl(BlobSerializer&) const {
}

void TestStage::serializeDataImpl(BlobSerializer&) const {
}

TestModel::TestModel(const Model& model) : _model(model) {}

const Model& TestModel::getBaseModel() const {
    return _model;
}

const DataVector& TestModel::getInputs() const {
    return _inputs;
}

const DataVector& TestModel::getOutputs() const {
    return _outputs;
}

const StageVector& TestModel::getStages() const {
    return _stages;
}

void TestModel::createInputs(std::vector<DataDesc> descriptors) {
    const auto& inputDescs = descriptors.empty() ? std::vector<DataDesc>{DataDesc{}} : descriptors;
    const auto numInputs = inputDescs.size();

    _model->attrs().set<int>("numInputs", numInputs);
    _inputs.resize(numInputs);

    for (int i = 0; i < numInputs; ++i) {
        _inputs[i] = _model->addInputData(formatString("Input %d", i), inputDescs[i]);
    }
}

void TestModel::createOutputs(std::vector<DataDesc> descriptors) {
    const auto& outputDescs = descriptors.empty() ? std::vector<DataDesc>{DataDesc{}} : descriptors;
    const auto numOutputs = outputDescs.size();

    _model->attrs().set<int>("numOutputs", numOutputs);
    _outputs.resize(numOutputs);

    for (int i = 0; i < numOutputs; ++i) {
        _outputs[i] = _model->addOutputData(formatString("Output %d", i), outputDescs[i]);
    }
}

Stage TestModel::addStage(
        const std::vector<InputInfo>& curInputInfos,
        const std::vector<OutputInfo>& curOutputInfos,
        StageType stageType) {
    DataVector curInputs;
    for (const auto& info : curInputInfos) {
        if (info.type == InputType::Original) {
            curInputs.push_back(_inputs.at(info.originalInputInd));
        } else if (info.type == InputType::Constant) {
            curInputs.push_back(_model->addConstData(formatString("Const {} / {}", _stages.size(), curInputs.size()), info.desc));
        } else {
            curInputs.push_back(_stages.at(info.prevStageInd)->output(info.prevStageOutputInd));
        }
    }

    DataVector curOutputs;
    for (const auto& info : curOutputInfos) {
        if (info.type == OutputType::Original) {
            curOutputs.push_back(_outputs.at(info.originalOutputInd));
        } else {
            auto data = _model->addNewData(formatString("Data %d / %d", _stages.size(), curOutputs.size()), info.desc);
            data->setMemReqs(info.memReq);
            curOutputs.push_back(std::move(data));
        }
    }

    auto stage = _model->addNewStage<TestStage>(
            formatString("Stage %m%m%d", std::setw(2), std::setfill('0'), _stages.size()),
            stageType,
            nullptr,
            curInputs,
            curOutputs);
    stage->attrs().set<int>("test_ind", _stages.size());

    _stages.push_back(stage);

    return stage;
}
void TestModel::setStageDataOrderInfo(
        int stageInd,
        const InOutPortMap<DimsOrder>& inputInfo,
        const InOutPortMap<DimsOrder>& outputInfo) {
    if (!inputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_DataOrder_info", inputInfo);
    }
    if (!outputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_DataOrder_info", outputInfo);
    }
}

void TestModel::setStageStridesInfo(
        int stageInd,
        const InOutPortMap<StridesRequirement>& inputInfo,
        const InOutPortMap<StridesRequirement>& outputInfo) {
    if (!inputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_Strides_info", inputInfo);
    }
    if (!outputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_Strides_info", outputInfo);
    }
}

void TestModel::setStageBatchInfo(
        int stageInd,
        const InOutPortMap<BatchSupport>& inputInfo) {
    if (!inputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_Batch_info", inputInfo);
    }
}

template <class StageRange>
void checkStageTestInds(const StageRange& stageRange, std::initializer_list<int> expectedInds) {
    auto stageVector = toVector(stageRange);

    ASSERT_EQ(expectedInds.size(), stageVector.size());

    size_t stageInd = 0;
    for (auto expectedInd : expectedInds) {
        ASSERT_EQ(expectedInd, stageVector[stageInd]->attrs().template get<int>("test_ind"));
        ++stageInd;
    }
}

template <class StageRange>
void checkStageTestInds(const StageRange& stageRange, std::initializer_list<int> expectedInds, const std::function<void(const Stage&)>& extraCheck) {
    auto stageVector = toVector(stageRange);

    ASSERT_EQ(expectedInds.size(), stageVector.size());

    size_t stageInd = 0;
    for (auto expectedInd : expectedInds) {
        ASSERT_EQ(expectedInd, stageVector[stageInd]->attrs().template get<int>("test_ind"));
        ++stageInd;

        ASSERT_NO_FATAL_FAILURE(extraCheck(stageVector[stageInd]));
    }
}

bool checkExecutionOrder(const Model& model, const std::vector<int>& execOrder) {
    auto it = execOrder.begin();

    for (const auto& stage : model->getStages()) {
        if (it == execOrder.end()) {
            return true;
        }

        if (stage->id() == *it) {
            ++it;
        }
    }

    return it == execOrder.end();
}

void GraphTransformerTest::SetUp() {
    _log = std::make_shared<Logger>(
            "Test",
            LogLevel::Debug,
            consoleOutput());

    stageBuilder = std::make_shared<StageBuilder>();
    frontEnd = std::make_shared<FrontEnd>(stageBuilder, _mockCore);
    backEnd = std::make_shared<BackEnd>();
    passManager = std::make_shared<PassManager>(stageBuilder, backEnd);

    config = createConfiguration();
}

void GraphTransformerTest::TearDown() {
    for (const auto& model : _models) {
        backEnd->dumpModel(model);
    }

    if (compileEnvInitialized) {
        CompileEnv::free();
    }
}

void GraphTransformerTest::InitCompileEnv() {
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_FILE_NAME")) {
        config.set(InferenceEngine::MYRIAD_DUMP_INTERNAL_GRAPH_FILE_NAME, envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_DIRECTORY")) {
        config.set(InferenceEngine::MYRIAD_DUMP_ALL_PASSES_DIRECTORY, envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_ALL_PASSES")) {
        config.set(InferenceEngine::MYRIAD_DUMP_ALL_PASSES, std::stoi(envVar) != 0
            ? InferenceEngine::PluginConfigParams::YES : InferenceEngine::PluginConfigParams::NO);
    }

    CompileEnv::init(platform, config, _log);
    compileEnvInitialized = true;
}

namespace {

    std::atomic<int> g_counter(0);

}

Model GraphTransformerTest::CreateModel() {
    const auto& env = CompileEnv::get();

    auto unitTest = testing::UnitTest::GetInstance();
    IE_ASSERT(unitTest != nullptr);
    auto curTestInfo = unitTest->current_test_info();
    IE_ASSERT(curTestInfo != nullptr);

    auto model = std::make_shared<ModelObj>(
            formatString("%s/%s", curTestInfo->test_case_name(), curTestInfo->name()));
    model->attrs().set<int>("index", g_counter.fetch_add(1));
    model->attrs().set<Resources>("resources", env.resources);

    _models.push_back(model);

    return model;
}

TestModel GraphTransformerTest::CreateTestModel() {
    return TestModel(CreateModel());
}

PluginConfiguration createConfiguration() {
    PluginConfiguration configuration;
    configuration.registerOption<LogLevelOption>();
    configuration.registerOption<CopyOptimizationOption>();
    configuration.registerOption<ProtocolOption>();
    configuration.registerOption<PowerConfigOption>();
    configuration.registerOption<HwAccelerationOption>();
    configuration.registerOption<HwExtraSplitOption>();
    configuration.registerOption<HwPoolConvMergeOption>();
    configuration.registerOption<HwBlackListOption>();
    configuration.registerOption<HwInjectStagesOption>();
    configuration.registerOption<HwDilationOption>();
    configuration.registerOption<TilingCMXLimitKBOption>();
    configuration.registerOption<WatchdogIntervalOption>();
    configuration.registerOption<EnableReceivingTensorTimeOption>();
    configuration.registerOption<PerfReportModeOption>();
    configuration.registerOption<PerfCountOption>();
    configuration.registerOption<PackDataInCMXOption>();
    configuration.registerOption<NumberOfSHAVEsOption>();
    configuration.registerOption<NumberOfCMXSlicesOption>();
    configuration.registerOption<ThroughputStreamsOption>();
    configuration.registerOption<IRWithScalesDirectoryOption>();
    configuration.registerOption<TensorStridesOption>();
    configuration.registerOption<IgnoreUnknownLayersOption>();
    configuration.registerOption<ForcePureTensorIteratorOption>();
    configuration.registerOption<EnableTensorIteratorUnrollingOption>();
    configuration.registerOption<ExclusiveAsyncRequestsOption>();
    configuration.registerOption<EnableWeightsAnalysisOption>();
    configuration.registerOption<EnableReplWithSCReluOption>();
    configuration.registerOption<EnablePermuteMergingOption>();
    configuration.registerOption<EnableMemoryTypesAnnotationOption>();
    configuration.registerOption<DumpInternalGraphFileNameOption>();
    configuration.registerOption<DumpAllPassesDirectoryOption>();
    configuration.registerOption<DumpAllPassesOption>();
    configuration.registerOption<DeviceIDOption>();
    configuration.registerOption<DeviceConnectTimeoutOption>();
    configuration.registerOption<DetectNetworkBatchOption>();
    configuration.registerOption<CustomLayersOption>();
    configuration.registerOption<ConfigFileOption>();
    configuration.registerOption<MemoryTypeOption>();
    configuration.registerOption<EnableForceResetOption>();
    configuration.registerOption<CheckPreprocessingInsideModelOption>();
    configuration.registerOption<EnableEarlyEltwiseReluFusionOption>();
    configuration.registerOption<EnableCustomReshapeParamOption>();
    configuration.registerOption<NoneLayersOption>();
    configuration.registerOption<EnableAsyncDMAOption>();

IE_SUPPRESS_DEPRECATED_START
    configuration.registerDeprecatedOption<DisableConvertStagesOption>(InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES);
    configuration.registerDeprecatedOption<DisableReorderOption>(InferenceEngine::MYRIAD_DISABLE_REORDER);
    configuration.registerDeprecatedOption<LogLevelOption>(VPU_CONFIG_KEY(LOG_LEVEL));
    configuration.registerDeprecatedOption<ProtocolOption>(VPU_MYRIAD_CONFIG_KEY(PROTOCOL));
    configuration.registerDeprecatedOption<HwAccelerationOption>(VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION));
    configuration.registerDeprecatedOption<EnableReceivingTensorTimeOption>(VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME));
    configuration.registerDeprecatedOption<DetectNetworkBatchOption>(VPU_CONFIG_KEY(DETECT_NETWORK_BATCH));
    configuration.registerDeprecatedOption<CustomLayersOption>(VPU_CONFIG_KEY(CUSTOM_LAYERS));
    configuration.registerDeprecatedOption<MemoryTypeOption>(VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE));
    configuration.registerDeprecatedOption<EnableForceResetOption>(VPU_MYRIAD_CONFIG_KEY(FORCE_RESET));
    configuration.registerDeprecatedOption<PlatformOption>(VPU_MYRIAD_CONFIG_KEY(PLATFORM));
IE_SUPPRESS_DEPRECATED_END

    return configuration;
}

} // namespace vpu
