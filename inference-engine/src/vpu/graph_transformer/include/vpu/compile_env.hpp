// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/graph_transformer.hpp>
#include <vpu/model/model.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/profiling.hpp>

namespace vpu {

struct DeviceResources {
    static int numShaves(const Platform& platform);
    static int numSlices(const Platform& platform);
    static int numStreams();
};

struct DefaultAllocation {
    static int numStreams(const Platform& platform, const CompilationConfig& configuration);
    static int numSlices(const Platform& platform, int numStreams);
    static int numShaves(const Platform& platform, int numStreams, int numSlices);
    static int tilingCMXLimit(int numSlices);
};

struct CompileEnv final {
public:
    Platform platform;
    Resources resources;

    CompilationConfig config;

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
            Platform platform,
            const CompilationConfig& config,
            const Logger::Ptr& log);
    static void updateConfig(const CompilationConfig& config);
    static void free();

private:
    explicit CompileEnv(Platform platform);
};

}  // namespace vpu
