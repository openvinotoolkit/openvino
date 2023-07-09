// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.hpp"

#include "ie/ie_plugin_config.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::hetero;

Configuration::Configuration() : 
    dump_graph(false),
    exclusive_async_requests(true),
    device_properties({ov::exclusive_async_requests(exclusive_async_requests)}) {}

Configuration::Configuration(const ov::AnyMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;

    for (const auto& it : config) {
        const auto& key = it.first;
        const auto& value = it.second;

        if (HETERO_CONFIG_KEY(DUMP_GRAPH_DOT) == key) {
            dump_graph = value.as<bool>();
        } else if ("TARGET_FALLBACK" == key || ov::device::priorities == key) {
            device_priorities = value.as<std::string>();
        } else if (ov::exclusive_async_requests == key) {
            exclusive_async_requests = value.as<bool>();
            // property should be passed to underlying devices as part of `GetDeviceConfig()`
            device_properties.emplace(key, value);
        } else {
            if (throwOnUnsupported)
                OPENVINO_THROW("Property was not found: ", key);
            device_properties.emplace(key, value);
        }
    }
}

ov::Any Configuration::Get(const std::string& name) const {
    if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)) {
        return {dump_graph};
    } else if (name == "TARGET_FALLBACK" || name == ov::device::priorities) {
        return {device_priorities};
    } else if (name == ov::exclusive_async_requests) {
        return {exclusive_async_requests};
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
}

std::vector<ov::PropertyName> Configuration::GetSupported() const {
    return {HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), "TARGET_FALLBACK", ov::device::priorities, ov::exclusive_async_requests};
}

ov::AnyMap Configuration::GetHeteroProperties() const {
    return {{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), dump_graph},
            {"TARGET_FALLBACK", device_priorities},
            {ov::device::priorities.name(), device_priorities},
            {ov::exclusive_async_requests.name(), exclusive_async_requests}};
}

ov::AnyMap Configuration::GetDeviceProperties() const {
    return device_properties;
}