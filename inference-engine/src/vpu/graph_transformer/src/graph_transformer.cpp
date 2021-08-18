// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/graph_transformer.hpp>
#include <vpu/graph_transformer_internal.hpp>

#include <climits>
#include <cstring>

#include <string>
#include <memory>
#include <list>
#include <vector>
#include <array>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <algorithm>
#include <map>
#include <streambuf>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <atomic>

#include <precision_utils.h>
#include <legacy/graph_tools.hpp>
#include <description_buffer.hpp>
#include <xml_parse_utils.h>
#include <legacy/ie_util_internal.hpp>

#include <vpu/vpu_config.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/stage_builder.hpp>
#include <vpu/frontend/frontend.hpp>
#include <vpu/backend/backend.hpp>
#include <vpu/middleend/pass_manager.hpp>
#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/file_system.hpp>
#include <vpu/utils/error.hpp>
#include <mvnc.h>

#include <vpu/configuration/options/hw_acceleration.hpp>
#include <vpu/configuration/options/tiling_cmx_limit_kb.hpp>
#include <vpu/configuration/options/number_of_shaves.hpp>
#include <vpu/configuration/options/throughput_streams.hpp>
#include <vpu/configuration/options/number_of_cmx_slices.hpp>
#include <vpu/configuration/options/ir_with_scales_directory.hpp>

namespace vpu {

//
// CompileEnv
//

namespace  {

thread_local CompileEnv* g_compileEnv = nullptr;

}  // namespace

CompileEnv::CompileEnv(ncDevicePlatform_t platform) : platform(platform) {}

const CompileEnv& CompileEnv::get() {
    IE_ASSERT(g_compileEnv != nullptr);
    IE_ASSERT(g_compileEnv->initialized);

    return *g_compileEnv;
}

const CompileEnv* CompileEnv::getOrNull() {
    IE_ASSERT(g_compileEnv == nullptr || g_compileEnv->initialized);

    return g_compileEnv;
}

void CompileEnv::init(ncDevicePlatform_t platform, const PluginConfiguration& config, const Logger::Ptr& log) {
    g_compileEnv = new CompileEnv(platform);
    g_compileEnv->config = config;
    g_compileEnv->log = log;

#ifdef ENABLE_PROFILING_RAW
    g_compileEnv->profile.setLogger(log);
#endif

    if (platform == ncDevicePlatform_t::NC_MYRIAD_2) {
        g_compileEnv->config.set(ie::MYRIAD_ENABLE_HW_ACCELERATION, ie::PluginConfigParams::NO);
    }

    const auto numExecutors = config.get<ThroughputStreamsOption>().hasValue()
        ? config.get<ThroughputStreamsOption>().get() : DefaultAllocation::numStreams(platform, config);
    VPU_THROW_UNLESS(numExecutors >= 1 && numExecutors <= DeviceResources::numStreams(),
        R"(Value of configuration option ("{}") must be in the range [{}, {}], actual is "{}")",
        ThroughputStreamsOption::key(), 1, DeviceResources::numStreams(), numExecutors);

    const auto numSlices  = config.get<NumberOfCMXSlicesOption>().hasValue()
        ? config.get<NumberOfCMXSlicesOption>().get()
        : DefaultAllocation::numSlices(platform, numExecutors);
    VPU_THROW_UNLESS(numSlices >= 1 && numSlices <= DeviceResources::numSlices(platform),
        R"(Value of configuration option ("{}") must be in the range [{}, {}], actual is "{}")",
        NumberOfCMXSlicesOption::key(), 1, DeviceResources::numSlices(platform), numSlices);

    int defaultCmxLimit = DefaultAllocation::tilingCMXLimit(numSlices);
    const auto tilingCMXLimit  = config.get<TilingCMXLimitKBOption>().hasValue()
        ? std::min<int>(config.get<TilingCMXLimitKBOption>().get() * 1024, defaultCmxLimit)
        : defaultCmxLimit;
    VPU_THROW_UNLESS(tilingCMXLimit >= 0,
        R"(Value of configuration option ("{}") must be greater than {}, actual is "{}")",
        TilingCMXLimitKBOption::key(), 0, tilingCMXLimit);

    const auto numShaves = config.get<NumberOfSHAVEsOption>().hasValue()
        ? config.get<NumberOfSHAVEsOption>().get()
        : DefaultAllocation::numShaves(platform, numExecutors, numSlices);
    VPU_THROW_UNLESS(numShaves >= 1 && numShaves <= DeviceResources::numShaves(platform),
        R"(Value of configuration option ("{}") must be in the range [{}, {}], actual is "{}")",
        NumberOfSHAVEsOption::key(), 1, DeviceResources::numShaves(platform), numShaves);

    const auto numAllocatedShaves = numShaves * numExecutors;
    VPU_THROW_UNLESS(numAllocatedShaves >= 1 && numAllocatedShaves <= DeviceResources::numShaves(platform),
        R"(Cannot allocate "{}" shaves: only {} is available)", numAllocatedShaves, DeviceResources::numShaves(platform));

    const auto numAllocatedSlices = numSlices * numExecutors;
    VPU_THROW_UNLESS(numAllocatedSlices >= 1 && numAllocatedSlices <= DeviceResources::numSlices(platform),
        R"(Cannot allocate "{}" slices: only {} is available)", numAllocatedSlices, DeviceResources::numSlices(platform));

    g_compileEnv->resources.numSHAVEs = numShaves;
    g_compileEnv->resources.numCMXSlices = numSlices;
    g_compileEnv->resources.numExecutors = numExecutors;
    g_compileEnv->resources.tilingCMXLimit = tilingCMXLimit;
    g_compileEnv->initialized = true;
}

void CompileEnv::updateConfig(const PluginConfiguration& config) {
    IE_ASSERT(g_compileEnv != nullptr);
    IE_ASSERT(g_compileEnv->initialized);

    g_compileEnv->config = config;
}

void CompileEnv::free() {
    IE_ASSERT(g_compileEnv != nullptr);
    IE_ASSERT(g_compileEnv->initialized);

    delete g_compileEnv;
    g_compileEnv = nullptr;
}

//
// compileNetwork
//

namespace {

CompiledGraph::Ptr compileImpl(const ie::CNNNetwork& network, const std::shared_ptr<ie::ICore> core) {
    const auto& env = CompileEnv::get();

    env.log->debug("Compile network [%s]", network.getName());
    VPU_LOGGER_SECTION(env.log);

    auto stageBuilder = std::make_shared<StageBuilder>();
    auto frontEnd = std::make_shared<FrontEnd>(stageBuilder, core);
    auto backEnd = std::make_shared<BackEnd>();
    auto passManager = std::make_shared<PassManager>(stageBuilder, backEnd);

    auto middleEnd = passManager->buildMiddleEnd();

    auto model = frontEnd->buildInitialModel(network);

    AutoScope autoDumper([backEnd, model]() {
        backEnd->dumpModel(model);
    });

    middleEnd->run(model);

    if (!env.config.get<IRWithScalesDirectoryOption>().empty()) {
        network.serialize(env.config.get<IRWithScalesDirectoryOption>() + "/" + network.getName() + "_scales.xml",
                          env.config.get<IRWithScalesDirectoryOption>() + "/" + network.getName() + "_scales.bin");
    }

    return backEnd->build(model, frontEnd->origLayers());
}

CompiledGraph::Ptr compileImpl(const Model& model) {
    auto stageBuilder = std::make_shared<StageBuilder>();
    auto backEnd      = std::make_shared<BackEnd>();
    auto passManager  = std::make_shared<PassManager>(stageBuilder, backEnd);

    auto middleEnd = passManager->buildMiddleEnd();

    AutoScope autoDumper([backEnd, model]() {
        backEnd->dumpModel(model);
    });

    middleEnd->run(model);

    return backEnd->build(model, {});
}

}  // namespace

CompiledGraph::Ptr compileNetwork(const ie::CNNNetwork& network, ncDevicePlatform_t platform, const PluginConfiguration& config, const Logger::Ptr& log,
                                  const std::shared_ptr<ie::ICore> core) {
    CompileEnv::init(platform, config, log);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    VPU_PROFILE(compileNetwork);

    return compileImpl(network, core);
}

CompiledGraph::Ptr compileModel(
        const Model& model,
        ncDevicePlatform_t platform,
        const PluginConfiguration& config,
        const Logger::Ptr& log) {
    CompileEnv::init(platform, config, log);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    VPU_PROFILE(compileModel);

    return compileImpl(model);
}

CompiledGraph::Ptr compileSubNetwork(const ie::CNNNetwork& network, const PluginConfiguration& subConfig, const std::shared_ptr<ie::ICore> core) {
    VPU_PROFILE(compileSubNetwork);

    const auto& env = CompileEnv::get();

    auto prevConfig = env.config;
    AutoScope autoRecover([prevConfig]() {
        CompileEnv::updateConfig(prevConfig);
    });

    CompileEnv::updateConfig(subConfig);

    return compileImpl(network, core);
}

//
// getSupportedLayers
//

std::set<std::string> getSupportedLayers(
    const ie::CNNNetwork& network,
    ncDevicePlatform_t platform,
    const PluginConfiguration& config,
    const Logger::Ptr& log,
    const std::shared_ptr<ie::ICore> core) {
    CompileEnv::init(platform, config, log);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    VPU_PROFILE(getSupportedLayers);

    auto stageBuilder = std::make_shared<StageBuilder>();
    auto frontEnd = std::make_shared<FrontEnd>(stageBuilder, core);
    return frontEnd->checkSupportedLayers(network);
}

int DeviceResources::numShaves(const ncDevicePlatform_t& platform) {
    return platform == ncDevicePlatform_t::NC_MYRIAD_2 ? 12 : 16;
}

int DeviceResources::numSlices(const ncDevicePlatform_t& platform) {
    return platform == ncDevicePlatform_t::NC_MYRIAD_2 ? 12 : 19;
}

int DeviceResources::numStreams() {
    return 3;
}

int DefaultAllocation::numStreams(const ncDevicePlatform_t& platform, const PluginConfiguration& configuration) {
    return platform == ncDevicePlatform_t::NC_MYRIAD_X && configuration.get<HwAccelerationOption>() ? 2 : 1;
}

int DefaultAllocation::numSlices(const ncDevicePlatform_t& platform, int numStreams) {
    const auto capabilities = DeviceResources::numSlices(platform);
    return capabilities / numStreams;
}

int DefaultAllocation::numShaves(const ncDevicePlatform_t& platform, int numStreams, int numSlices) {
    const auto numAvailableShaves = DeviceResources::numShaves(platform);
    if (numStreams == 1) {
        return numAvailableShaves;
    }

    const auto numAllocatedSlices = numStreams * numSlices;
    VPU_THROW_UNLESS(numAllocatedSlices >= numAvailableShaves,
        R"(Number of allocated slices in default mode must be not less than number of available shaves, but {} < {} are provided)", numAllocatedSlices,
        numAvailableShaves);

    // each shave must have corresponding slice
    // there are cases when number of available slices more than available shaves (e.g. Myriad-X: 19 slices and 16 shaves)
    // allocated shaves and slices must be in a continuous range (e.g. allocated slices 0-8 and shaves 0-6)
    // conditions above lead to unused shaves during allocation in some cases
    // e.g. slices: 0-8 and 9-17, shaves: 0-6 and 9-15, shaves 7 and 8 are unused
    const auto numUnusedShaves = numAllocatedSlices - numAvailableShaves;
    const auto numShavesForAllocation = numAvailableShaves - numUnusedShaves;
    return numShavesForAllocation / numStreams;
}

int DefaultAllocation::tilingCMXLimit(int numSlices) {
    return (numSlices / 2) * CMX_SLICE_SIZE + CMX_SLICE_SIZE / 2;
}

}  // namespace vpu
