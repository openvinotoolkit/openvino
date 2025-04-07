// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace ov {
namespace template_plugin {

// ! [configuration:header]

struct Configuration {
    Configuration();
    Configuration(const Configuration&) = default;
    Configuration(Configuration&&) = default;
    Configuration& operator=(const Configuration&) = default;
    Configuration& operator=(Configuration&&) = default;

    explicit Configuration(const ov::AnyMap& config,
                           const Configuration& defaultCfg = {},
                           const bool throwOnUnsupported = true);

    ov::Any Get(const std::string& name) const;

    // Plugin configuration parameters

    int device_id = 0;
    bool perf_count = false;
    ov::threading::IStreamsExecutor::Config streams_executor_config{};
    int streams = 1;
    int threads = 0;
    int threads_per_stream = 0;
    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::LATENCY;
    uint32_t num_requests = 1;
    bool disable_transformations = false;
    bool exclusive_async_requests = false;

    // unused
    ov::element::Type inference_precision = ov::element::dynamic;
    ov::hint::ExecutionMode execution_mode = ov::hint::ExecutionMode::ACCURACY;
    ov::log::Level log_level = ov::log::Level::NO;

    ov::hint::Priority model_priority = ov::hint::Priority::DEFAULT;

    EncryptionCallbacks encryption_callbacks{};
    std::filesystem::path weights_path{};
    AnyMap compiled_model_runtime_properties{};
    CacheMode cache_mode{CacheMode::OPTIMIZE_SPEED};
};
// ! [configuration:header]

}  // namespace template_plugin
}  // namespace ov
