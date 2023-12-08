// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.hpp"

#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "template/properties.hpp"

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

        if (ov::template_plugin::disable_transformations == key) {
            disable_transformations = value.as<bool>();
        } else if (ov::internal::exclusive_async_requests == key) {
            exclusive_async_requests = value.as<bool>();
        } else if (streamExecutorConfigKeys.end() !=
                   std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            streams_executor_config.set_property(key, value);
        } else if (ov::device::id == key) {
            device_id = std::stoi(value.as<std::string>());
            OPENVINO_ASSERT(device_id <= 0, "Device ID ", device_id, " is not supported");
        } else if (ov::enable_profiling == key) {
            perf_count = value.as<bool>();
        } else if (ov::hint::performance_mode == key) {
            std::stringstream strm{value.as<std::string>()};
            strm >> performance_mode;
        } else if (ov::hint::inference_precision == key) {
            inference_precision = value.as<ov::element::Type>();
        } else if (ov::hint::execution_mode == key) {
            execution_mode = value.as<ov::hint::ExecutionMode>();
            if ((execution_mode != ov::hint::ExecutionMode::ACCURACY) &&
                (execution_mode != ov::hint::ExecutionMode::PERFORMANCE)) {
                OPENVINO_THROW("Unsupported execution mode, should be ACCURACY or PERFORMANCE, but was: ",
                               value.as<std::string>());
            }
        } else if (ov::num_streams == key) {
            streams_executor_config.set_property(key, value);
        } else if (ov::hint::num_requests == key) {
            auto tmp_val = value.as<std::string>();
            int tmp_i = std::stoi(tmp_val);
            if (tmp_i >= 0)
                num_requests = tmp_i;
            else
                OPENVINO_THROW("Incorrect value, it should be unsigned integer: ", key);
        } else if (ov::log::level == key) {
            log_level = value.as<ov::log::Level>();
        } else if (ov::hint::model_priority == key) {
            model_priority = value.as<ov::hint::Priority>();
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
    } else if (name == ov::device::id) {
        return {std::to_string(device_id)};
    } else if (name == ov::enable_profiling) {
        return {perf_count};
    } else if (name == ov::internal::exclusive_async_requests) {
        return {exclusive_async_requests};
    } else if (name == ov::template_plugin::disable_transformations) {
        return {disable_transformations};
    } else if (name == ov::num_streams) {
        return {std::to_string(streams_executor_config._streams)};
    } else if (name == ov::internal::cpu_bind_thread) {
        return streams_executor_config.get_property(name);
    } else if (name == ov::inference_num_threads) {
        return {std::to_string(streams_executor_config._threads)};
    } else if (name == ov::internal::threads_per_stream) {
        return {std::to_string(streams_executor_config._threadsPerStream)};
    } else if (name == ov::hint::performance_mode) {
        return performance_mode;
    } else if (name == ov::hint::inference_precision) {
        return inference_precision;
    } else if (name == ov::hint::execution_mode) {
        return execution_mode;
    } else if (name == ov::hint::num_requests) {
        return num_requests;
    } else if (name == ov::log::level) {
        return log_level;
    } else if (name == ov::hint::model_priority) {
        return model_priority;
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
}
