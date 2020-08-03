// Copyright (C) 2018-2020 Intel Corporation
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
#include <graph_tools.hpp>
#include <description_buffer.hpp>
#include <xml_parse_utils.h>
#include <ie_util_internal.hpp>

#include <vpu/parsed_config.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/stage_builder.hpp>
#include <vpu/frontend/frontend.hpp>
#include <vpu/backend/backend.hpp>
#include <vpu/middleend/pass_manager.hpp>
#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/file_system.hpp>

namespace vpu {

//
// CompileEnv
//

namespace  {

thread_local CompileEnv* g_compileEnv = nullptr;

}  // namespace

CompileEnv::CompileEnv(Platform platform) : platform(platform) {}

const CompileEnv& CompileEnv::get() {
    IE_ASSERT(g_compileEnv != nullptr);
    IE_ASSERT(g_compileEnv->initialized);

    return *g_compileEnv;
}

const CompileEnv* CompileEnv::getOrNull() {
    IE_ASSERT(g_compileEnv == nullptr || g_compileEnv->initialized);

    return g_compileEnv;
}

void CompileEnv::init(Platform platform, const CompilationConfig& config, const Logger::Ptr& log) {
    g_compileEnv = new CompileEnv(platform);
    g_compileEnv->config = config;
    g_compileEnv->log = log;

#ifdef ENABLE_PROFILING_RAW
    g_compileEnv->profile.setLogger(log);
#endif

    if (platform == Platform::MYRIAD_2) {
        g_compileEnv->config.hwOptimization = false;
    }

    VPU_THROW_UNLESS(g_compileEnv->config.numSHAVEs <= g_compileEnv->config.numCMXSlices,
        R"(Value of configuration option ("{}") must be not greater than value of configuration option ("{}"), but {} > {} are provided)",
        VPU_CONFIG_KEY(NUMBER_OF_SHAVES), VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES), config.numSHAVEs, config.numCMXSlices);

    const auto numExecutors = config.numExecutors != -1 ? config.numExecutors : DefaultAllocation::numStreams(platform, config);
    VPU_THROW_UNLESS(numExecutors >= 1 && numExecutors <= DeviceResources::numStreams(),
        R"(Value of configuration option ("{}") must be in the range [{}, {}], actual is "{}")",
        VPU_MYRIAD_CONFIG_KEY(THROUGHPUT_STREAMS), 1, DeviceResources::numStreams(), numExecutors);

    const auto numSlices  = config.numCMXSlices != -1 ? config.numCMXSlices : DefaultAllocation::numSlices(platform, numExecutors);
    VPU_THROW_UNLESS(numSlices >= 1 && numSlices <= DeviceResources::numSlices(platform),
        R"(Value of configuration option ("{}") must be in the range [{}, {}], actual is "{}")",
        VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES), 1, DeviceResources::numSlices(platform), numSlices);

    int defaultCmxLimit = DefaultAllocation::tilingCMXLimit(numSlices);
    const auto tilingCMXLimit  = config.tilingCMXLimitKB != -1 ? std::min(config.tilingCMXLimitKB * 1024, defaultCmxLimit) : defaultCmxLimit;
    VPU_THROW_UNLESS(tilingCMXLimit >= 0,
        R"(Value of configuration option ("{}") must be greater than {}, actual is "{}")",
        VPU_CONFIG_KEY(TILING_CMX_LIMIT_KB), 0, tilingCMXLimit);

    const auto numShaves = config.numSHAVEs != -1 ? config.numSHAVEs : DefaultAllocation::numShaves(platform, numExecutors, numSlices);
    VPU_THROW_UNLESS(numShaves >= 1 && numShaves <= DeviceResources::numShaves(platform),
        R"(Value of configuration option ("{}") must be in the range [{}, {}], actual is "{}")",
        VPU_CONFIG_KEY(NUMBER_OF_SHAVES), 1, DeviceResources::numShaves(platform), numShaves);

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

void CompileEnv::updateConfig(const CompilationConfig& config) {
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

CompiledGraph::Ptr compileImpl(ie::ICNNNetwork& network,
                               const ie::ICore* core) {
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

    if (!env.config.irWithVpuScalesDir.empty()) {
        network.serialize(env.config.irWithVpuScalesDir + "/" + network.getName() + "_scales.xml",
                          env.config.irWithVpuScalesDir + "/" + network.getName() + "_scales.bin",
                          nullptr);
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

CompiledGraph::Ptr compileNetwork(
        ie::ICNNNetwork& network,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log,
        const ie::ICore* core) {
    CompileEnv::init(platform, config, log);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    VPU_PROFILE(compileNetwork);

    return compileImpl(network, core);
}

CompiledGraph::Ptr compileModel(
        const Model& model,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log) {
    CompileEnv::init(platform, config, log);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    VPU_PROFILE(compileModel);

    return compileImpl(model);
}

CompiledGraph::Ptr compileSubNetwork(
        ie::ICNNNetwork& network,
        const CompilationConfig& subConfig,
        const ie::ICore* core) {
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
        const ie::ICNNNetwork& network,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log,
        const ie::ICore* core) {
    CompileEnv::init(platform, config, log);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    VPU_PROFILE(getSupportedLayers);

    auto stageBuilder = std::make_shared<StageBuilder>();
    auto frontEnd = std::make_shared<FrontEnd>(stageBuilder, core);

    auto clonedNetworkImpl = ie::cloneNet(network);

    return frontEnd->checkSupportedLayers(*clonedNetworkImpl);
}

int DeviceResources::numShaves(const Platform& platform) {
    return platform == Platform::MYRIAD_2 ? 12 : 16;
}

int DeviceResources::numSlices(const Platform& platform) {
    return platform == Platform::MYRIAD_2 ? 12 : 19;
}

int DeviceResources::numStreams() {
    return 3;
}

int DefaultAllocation::numStreams(const Platform& platform, const CompilationConfig& configuration) {
    return platform == Platform::MYRIAD_X && configuration.hwOptimization ? 2 : 1;
}

int DefaultAllocation::numSlices(const Platform& platform, int numStreams) {
    const auto capabilities = DeviceResources::numSlices(platform);
    return capabilities / numStreams;
}

int DefaultAllocation::numShaves(const Platform& platform, int numStreams, int numSlices) {
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
