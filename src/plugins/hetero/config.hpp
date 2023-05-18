// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace ov {
namespace hetero_plugin {

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

    ov::AnyMap GetDeviceConfig() const;

    // Plugin configuration parameters

    bool dump_graph = false;
    bool exclusive_async_requests = true;
    std::string device_priorities;
};
// ! [configuration:header]

}  // namespace hetero_plugin
}  // namespace ov
