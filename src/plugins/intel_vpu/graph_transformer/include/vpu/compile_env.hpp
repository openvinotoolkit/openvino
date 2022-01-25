// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/graph_transformer.hpp>
#include <vpu/model/model.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/profiling.hpp>
#include <mvnc.h>

namespace vpu {

struct DeviceResources {
    static int numShaves();
    static int numSlices();
    static int numStreams();
};

struct DefaultAllocation {
    static int numStreams(const PluginConfiguration& configuration);
    static int numSlices(int numStreams);
    static int numShaves(int numStreams, int numSlices);
    static int tilingCMXLimit(int numSlices);
};

struct CompileEnv final {
public:
    Resources resources;

    PluginConfiguration config;

    Logger::Ptr log;

#ifdef ENABLE_PROFILING_RAW
    mutable Profiler profile;
#endif

    bool initialized = false;

    CompileEnv(const CompileEnv&) = delete;
    CompileEnv& operator=(const CompileEnv&) = delete;

    CompileEnv(CompileEnv&&) = delete;
    CompileEnv& operator=(CompileEnv&&) = delete;

    static const CompileEnv& get();
    static const CompileEnv* getOrNull();

    static void init(
        const PluginConfiguration& config,
        const Logger::Ptr& log);
    static void updateConfig(const PluginConfiguration& config);
    static void free();

private:
    CompileEnv();
};

}  // namespace vpu
