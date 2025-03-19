// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/runtime/properties.hpp"
#include "properties.hpp"

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

    ov::Any get(const std::string& name) const;

    std::vector<ov::PropertyName> get_supported() const;

    ov::AnyMap get_hetero_properties() const;

    ov::AnyMap get_device_properties() const;

    bool dump_dot_files() const;

    std::string device_priorities;

    std::set<ov::hint::ModelDistributionPolicy> modelDistributionPolicy = {};

    EncryptionCallbacks encryption_callbacks;

    ov::AnyMap device_properties;
};
}  // namespace hetero
}  // namespace ov