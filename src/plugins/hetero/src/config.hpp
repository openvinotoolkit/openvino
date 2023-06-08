// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace ov {
namespace hetero {
struct Configuration {
    Configuration();
    Configuration(const Configuration&) = default;
    Configuration(Configuration&&) = default;
    Configuration& operator=(const Configuration&) = default;
    Configuration& operator=(Configuration&&) = default;

    explicit Configuration(ov::AnyMap& config, const Configuration& defaultCfg = {}, bool throwOnUnsupported = false);

    ov::Any Get(const std::string& name) const;
    std::vector<ov::PropertyName> GetSupported() const;

    ov::AnyMap GetHeteroConfig() const;
    ov::AnyMap GetDeviceConfig() const;

    // Plugin configuration parameters

    bool dump_graph = false;
    bool exclusive_async_requests = true;
    std::string device_priorities;
};
}  // namespace hetero
}  // namespace ov