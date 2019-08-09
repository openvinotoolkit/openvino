// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/graph_transformer.hpp>

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
#include <details/caseless.hpp>
#include <graph_tools.hpp>
#include <description_buffer.hpp>
#include <xml_parse_utils.h>

#include <vpu/parsed_config.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/frontend/stage_builder.hpp>
#include <vpu/frontend/frontend.hpp>
#include <vpu/pass_manager.hpp>
#include <vpu/backend/backend.hpp>
#include <vpu/allocator.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/file_system.hpp>

namespace vpu {

//
// CompileEnv
//

namespace  {

thread_local CompileEnv *g_compileEnv = nullptr;

}  // namespace

const CompileEnv& CompileEnv::get() {
    IE_ASSERT(g_compileEnv != nullptr);
    IE_ASSERT(g_compileEnv->initialized);

    return *g_compileEnv;
}

const CompileEnv* CompileEnv::getOrNull() {
    IE_ASSERT(g_compileEnv == nullptr || g_compileEnv->initialized);

    return g_compileEnv;
}

void CompileEnv::init(
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log) {
    IE_ASSERT(g_compileEnv == nullptr);
    g_compileEnv = new CompileEnv();

    g_compileEnv->platform = platform;
    g_compileEnv->config = config;
    g_compileEnv->log = log;

#if ENABLE_PROFILING_RAW
    g_compileEnv->profile.setLogger(log);
#endif

    if (g_compileEnv->platform == Platform::MYRIAD_2) {
        g_compileEnv->config.hwOptimization = false;
    }

    if (g_compileEnv->config.numSHAVEs > g_compileEnv->config.numCMXSlices) {
        VPU_THROW_EXCEPTION
                << "Invalid config value for VPU_NUMBER_OF_SHAVES. "
                << "It is expected that the number of shaves is less than number of CMX slices";
    }

    if ((g_compileEnv->config.numSHAVEs == -1) && (g_compileEnv->config.numCMXSlices == -1)) {
        if (g_compileEnv->platform == Platform::MYRIAD_2) {
            g_compileEnv->resources.numCMXSlices = 12;
            g_compileEnv->resources.numSHAVEs = 12;
            g_compileEnv->resources.cmxLimit = 0;
        } else {
            if (g_compileEnv->config.hwOptimization) {
                g_compileEnv->resources.numCMXSlices = 9;
                g_compileEnv->resources.numSHAVEs = 7;
                g_compileEnv->resources.cmxLimit = (g_compileEnv->resources.numCMXSlices / 2) * CMX_SLICE_SIZE + CMX_SLICE_SIZE / 2;
            } else {
                g_compileEnv->resources.numCMXSlices = 16;
                g_compileEnv->resources.numSHAVEs = 16;
                g_compileEnv->resources.cmxLimit = 0;
            }
        }
    } else {
        if (g_compileEnv->platform == Platform::MYRIAD_2) {
            if ((g_compileEnv->config.numSHAVEs > 12) || (g_compileEnv->config.numSHAVEs < 1)) {
                VPU_THROW_EXCEPTION
                    << "Number of SHAVES should be in the range of 1 .. 12";
            }

            g_compileEnv->resources.numCMXSlices = g_compileEnv->config.numCMXSlices;
            g_compileEnv->resources.numSHAVEs = g_compileEnv->config.numSHAVEs;
            g_compileEnv->resources.cmxLimit = 0;
        } else {
            if ((g_compileEnv->config.numSHAVEs > 16) || (g_compileEnv->config.numSHAVEs < 1)) {
                VPU_THROW_EXCEPTION
                    << "Number of SHAVES should be in the range of 1 .. 16";
            }

            g_compileEnv->resources.numCMXSlices = g_compileEnv->config.numCMXSlices;
            g_compileEnv->resources.numSHAVEs = g_compileEnv->config.numSHAVEs;
            g_compileEnv->resources.cmxLimit = (g_compileEnv->resources.numCMXSlices / 2) * CMX_SLICE_SIZE + CMX_SLICE_SIZE / 2;
        }
    }

    g_compileEnv->netConfig.parse(g_compileEnv->config);

    if (g_compileEnv->netConfig.hasManualDataScale()) {
        g_compileEnv->config.hwAdaptiveMode = false;
    }

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

CompiledGraph::Ptr compileImpl(const ie::ICNNNetwork& network) {
    const auto& env = CompileEnv::get();

    env.log->debug("Compile network [%s]", network.getName());
    VPU_LOGGER_SECTION(env.log);

    auto stageBuilder = std::make_shared<StageBuilder>();
    auto frontEnd = std::make_shared<FrontEnd>(stageBuilder);
    auto backEnd = std::make_shared<BackEnd>();
    auto passManager = std::make_shared<PassManager>(stageBuilder, backEnd);

    auto middleEnd = passManager->buildMiddleEnd();

    auto model = frontEnd->buildInitialModel(network);

    AutoScope autoDumper([backEnd, model]() {
        backEnd->dumpModel(model);
    });

    middleEnd->run(model);

    return backEnd->build(model, frontEnd->allLayers());
}

}  // namespace

CompiledGraph::Ptr compileNetwork(
        const ie::ICNNNetwork& network,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log) {
    CompileEnv::init(platform, config, log);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    VPU_PROFILE(compileNetwork);

    return compileImpl(network);
}

CompiledGraph::Ptr compileSubNetwork(
        const ie::ICNNNetwork& network,
        const CompilationConfig& subConfig) {
    VPU_PROFILE(compileSubNetwork);

    const auto& env = CompileEnv::get();

    auto prevConfig = env.config;
    AutoScope autoRecover([prevConfig]() {
        CompileEnv::updateConfig(prevConfig);
    });

    CompileEnv::updateConfig(subConfig);

    return compileImpl(network);
}

//
// getSupportedLayers
//

std::set<std::string> getSupportedLayers(
        const ie::ICNNNetwork& network,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log) {
    CompileEnv::init(platform, config, log);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    VPU_PROFILE(getSupportedLayers);

    auto stageBuilder = std::make_shared<StageBuilder>();
    auto frontEnd = std::make_shared<FrontEnd>(stageBuilder);

    return frontEnd->checkSupportedLayers(network);
}

}  // namespace vpu
