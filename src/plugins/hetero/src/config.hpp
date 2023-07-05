// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace hetero {
struct Configuration {
    Configuration();
    Configuration(const Configuration&) = default;
    Configuration(Configuration&&) = default;
    Configuration& operator=(const Configuration&) = default;
    Configuration& operator=(Configuration&&) = default;

    explicit Configuration(const ov::AnyMap& config,
                           const Configuration& defaultCfg = {},
                           bool throwOnUnsupported = false);

    ov::Any Get(const std::string& name) const;
    std::vector<ov::PropertyName> GetSupported() const;

    ov::AnyMap GetHeteroProperties() const;
    ov::AnyMap GetDeviceProperties() const;

    // Plugin configuration parameters

    bool dump_graph = false;
    bool exclusive_async_requests = true;
    std::string device_priorities;
    ov::AnyMap device_properties;
};
}  // namespace hetero
}  // namespace ov