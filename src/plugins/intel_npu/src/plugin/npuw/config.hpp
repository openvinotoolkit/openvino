// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <string>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace npuw {

const char *get_env(const std::vector<std::string> &list_to_try,
                    const char *def_val = nullptr);

struct Configuration {
    Configuration();
    Configuration(const Configuration&) = default;
    Configuration(Configuration&&) = default;
    Configuration& operator=(const Configuration&) = default;
    Configuration& operator=(Configuration&&) = default;

    explicit Configuration(const ov::AnyMap& config,
                           const Configuration& defaultCfg = {});

    ov::Any get(const std::string& name) const;

    std::vector<ov::PropertyName> get_supported() const;

    ov::AnyMap get_npuw_properties() const;

    ov::AnyMap get_device_properties() const;

    bool dump_graph;
    std::string device_priorities;
    ov::AnyMap device_properties;
};
}  // namespace npuw
}  // namespace ov
