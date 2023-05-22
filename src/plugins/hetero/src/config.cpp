// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.hpp"

// #include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <ie_plugin_config.hpp> // TODO vurusovs create file with custom properties

#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::hetero;

Configuration::Configuration() {}


Configuration::Configuration(const ov::AnyMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;

    for (auto&& c : config) {
        const auto& key = c.first;
        const auto& value = c.second;

        if (HETERO_CONFIG_KEY(DUMP_GRAPH_DOT) == key)   // TODO vurusovs: rewrite HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)
            dump_graph = value.as<bool>();
        else if (ov::device::priorities.name() == key)
            device_priorities = value.as<std::string>();
        else if (ov::exclusive_async_requests == key)
            exclusive_async_requests = value.as<bool>();
        else if (throwOnUnsupported) {
            OPENVINO_THROW("Property was not found: ", key);
        }
    }
}

ov::Any Configuration::Get(const std::string& name) const {
    if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)) {
        return {dump_graph};
    } else if (name == ov::device::priorities) {
        return {device_priorities};
    } else if (name == ov::exclusive_async_requests) {
        return {exclusive_async_requests};
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
}

ov::AnyMap Configuration::GetDeviceConfig() const {
    return {{ov::exclusive_async_requests.name(), exclusive_async_requests}};
}