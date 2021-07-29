// Copyright (C) 2018-2021 Intel Corporation
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
    static int numShaves(const ncDevicePlatform_t& platform);
    static int numSlices(const ncDevicePlatform_t& platform);
    static int numStreams();
};

struct DefaultAllocation {
    static int numStreams(const ncDevicePlatform_t& platform, const PluginConfiguration& configuration);
    static int numSlices(const ncDevicePlatform_t& platform, int numStreams);
    static int numShaves(const ncDevicePlatform_t& platform, int numStreams, int numSlices);
    static int tilingCMXLimit(int numSlices);
};

struct CompileEnv final {
public:
    ncDevicePlatform_t platform;
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
        ncDevicePlatform_t platform,
        const PluginConfiguration& config,
        const Logger::Ptr& log);
    static void updateConfig(const PluginConfiguration& config);
    static void free();

private:
    explicit CompileEnv(ncDevicePlatform_t platform);
};

}  // namespace vpu
