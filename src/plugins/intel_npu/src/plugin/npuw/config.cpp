// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.hpp"

#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::npuw;

const char* ov::npuw::get_env(const std::vector<std::string> &list_to_try,
                              const char *def_val) {
    for (auto &&key : list_to_try) {
        const char *pstr = std::getenv(key.c_str());
        if (pstr) return pstr;
    }
    return def_val;
}

Configuration::Configuration() : dump_graph(false) {}

Configuration::Configuration(const ov::AnyMap& config, const Configuration& defaultCfg) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    *this = defaultCfg;

    for (const auto& it : config) {
        const auto& key = it.first;
        const auto& value = it.second;

        if (ov::device::priorities == key) {
            // Use priorities to override the default order only
            device_priorities = value.as<std::string>();
        } else {
            device_properties.emplace(key, value);
        }
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::Any Configuration::get(const std::string& name) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (name == ov::device::priorities) {
        return {device_priorities};
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::vector<ov::PropertyName> Configuration::get_supported() const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    static const std::vector<ov::PropertyName> names = {
        // NPUW_CONFIG_KEY(DUMP_GRAPH_DOT),
        ov::device::priorities};
    return names;
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::AnyMap Configuration::get_npuw_properties() const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return {// {NPUW_CONFIG_KEY(DUMP_GRAPH_DOT), dump_graph},
            {ov::device::priorities.name(), device_priorities}};
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::AnyMap Configuration::get_device_properties() const {
    return device_properties;
}
