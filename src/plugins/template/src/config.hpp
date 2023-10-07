// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
    ov::threading::IStreamsExecutor::Config streams_executor_config;
    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::LATENCY;
    uint32_t num_requests = 1;
    bool disable_transformations = false;
    bool exclusive_async_requests = false;

    // unused
    ov::element::Type inference_precision = ov::element::undefined;
    ov::hint::ExecutionMode execution_mode = ov::hint::ExecutionMode::ACCURACY;
    ov::log::Level log_level = ov::log::Level::NO;

    ov::hint::Priority model_priority = ov::hint::Priority::DEFAULT;
};
// ! [configuration:header]

}  // namespace template_plugin
}  // namespace ov
