// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/graph_transformer.hpp>
#include <vpu/network_config.hpp>
#include <vpu/model/model.hpp>
#include <vpu/utils/logger.hpp>

namespace vpu {

struct CompileEnv final {
    Platform platform = Platform::UNKNOWN;
    Resources resources;

    CompilationConfig config;
    NetworkConfig netConfig;

    Logger::Ptr log;

    bool initialized = false;

    static const CompileEnv& get();

    static void init(
            Platform platform,
            const CompilationConfig& config,
            const Logger::Ptr& log);
    static void updateConfig(const CompilationConfig& config);
    static void free();
};

}  // namespace vpu
