// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.hpp"

#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <ie_plugin_config.hpp>

#include "template/config.hpp"

using namespace ov::template_plugin;

Configuration::Configuration() {}

Configuration::Configuration(const ov::AnyMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;
    // If plugin needs to use ov::threading::StreamsExecutor it should be able to process its configuration
    auto streamExecutorConfigKeys =
        streams_executor_config.get_property(ov::supported_properties.name()).as<std::vector<std::string>>();
    for (auto&& c : config) {
        const auto& key = c.first;
        const auto& value = c.second;

        if (ov::template_plugin::throughput_streams == key) {
            streams_executor_config.set_property(CONFIG_KEY(CPU_THROUGHPUT_STREAMS), value);
        } else if (streamExecutorConfigKeys.end() !=
                   std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            streams_executor_config.set_property(key, value);
        } else if (CONFIG_KEY(DEVICE_ID) == key) {
            device_id = std::stoi(value.as<std::string>());
            OPENVINO_ASSERT(device_id <= 0, "Device ID ", device_id, " is not supported");
        } else if (CONFIG_KEY(PERF_COUNT) == key) {
            perf_count = (CONFIG_VALUE(YES) == value.as<std::string>());
        } else if (ov::hint::performance_mode == key) {
            std::stringstream strm{value.as<std::string>()};
            strm >> performance_mode;
        } else if (throwOnUnsupported) {
            OPENVINO_THROW("Property was not found: ", key);
        }
    }
}

ov::Any Configuration::Get(const std::string& name) const {
    auto streamExecutorConfigKeys =
        streams_executor_config.get_property(ov::supported_properties.name()).as<std::vector<std::string>>();
    if ((streamExecutorConfigKeys.end() !=
         std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), name))) {
        return streams_executor_config.get_property(name);
    } else if (name == CONFIG_KEY(DEVICE_ID)) {
        return {std::to_string(device_id)};
    } else if (name == CONFIG_KEY(PERF_COUNT)) {
        return {perf_count};
    } else if (name == ov::template_plugin::throughput_streams || name == CONFIG_KEY(CPU_THROUGHPUT_STREAMS)) {
        return {std::to_string(streams_executor_config._streams)};
    } else if (name == CONFIG_KEY(CPU_BIND_THREAD)) {
        return streams_executor_config.get_property(name);
    } else if (name == CONFIG_KEY(CPU_THREADS_NUM)) {
        return {std::to_string(streams_executor_config._threads)};
    } else if (name == CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)) {
        return {std::to_string(streams_executor_config._threadsPerStream)};
    } else if (name == ov::hint::performance_mode) {
        return performance_mode;
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
}
