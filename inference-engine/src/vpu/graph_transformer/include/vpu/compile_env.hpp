// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/graph_transformer.hpp>
#include <vpu/model/model.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/profiling.hpp>

namespace vpu {

struct CompileEnv final {
public:
    Platform platform = Platform::UNKNOWN;
    Resources resources;

    CompilationConfig config;

    Logger::Ptr log;

#ifdef ENABLE_PROFILING_RAW
    mutable Profiler profile;
#endif

    bool initialized = false;

public:
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
    inline CompileEnv() = default;
};

}  // namespace vpu
