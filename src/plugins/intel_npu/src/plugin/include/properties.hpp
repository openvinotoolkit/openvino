// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/al/config/config.hpp"
#include "metrics.hpp"

namespace intel_npu {

enum class PropertiesType { PLUGIN, COMPILED_MODEL };

class Properties final {
public:
    Properties(const PropertiesType pType, Config& config, const std::shared_ptr<Metrics>& metrics = nullptr);

    void registerProperties();

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const;
    void set_property(const ov::AnyMap& properties);

private:
    PropertiesType _pType;
    Config& _config;
    std::shared_ptr<Metrics> _metrics;

    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>> _properties;
    std::vector<ov::PropertyName> _supportedProperties;
};

}  // namespace intel_npu