// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.hpp"

#include "ie/ie_plugin_config.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::hetero;

Configuration::Configuration() : dump_graph(false) {}

Configuration::Configuration(const ov::AnyMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    *this = defaultCfg;

    for (const auto& it : config) {
        const auto& key = it.first;
        const auto& value = it.second;

        if (HETERO_CONFIG_KEY(DUMP_GRAPH_DOT) == key) {
            dump_graph = value.as<bool>();
        } else if ("TARGET_FALLBACK" == key || ov::device::priorities == key) {
            device_priorities = value.as<std::string>();
        } else {
            if (throwOnUnsupported)
                OPENVINO_THROW("Property was not found: ", key);
            device_properties.emplace(key, value);
        }
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::Any Configuration::get(const std::string& name) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)) {
        return {dump_graph};
    } else if (name == "TARGET_FALLBACK" || name == ov::device::priorities) {
        return {device_priorities};
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::vector<ov::PropertyName> Configuration::get_supported() const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    static const std::vector<ov::PropertyName> names = {HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
                                                        "TARGET_FALLBACK",
                                                        ov::device::priorities};
    return names;
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::AnyMap Configuration::get_hetero_properties() const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return {{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), dump_graph},
            {"TARGET_FALLBACK", device_priorities},
            {ov::device::priorities.name(), device_priorities}};
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::AnyMap Configuration::get_device_properties() const {
    return device_properties;
}

bool Configuration::dump_dot_files() const {
    bool res = dump_graph;
    if (std::getenv("OPENVINO_HETERO_VISUALIZE"))
        res = true;
    return res;
}