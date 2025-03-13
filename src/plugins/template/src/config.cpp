// Copyright (C) 2018-2025 Intel Corporation
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
    for (auto&& [key, value] : config) {
        if (ov::template_plugin::disable_transformations == key) {
            disable_transformations = value.as<bool>();
        } else if (ov::internal::exclusive_async_requests == key) {
            exclusive_async_requests = value.as<bool>();
        } else if (ov::num_streams.name() == key) {
            ov::Any val = value.as<std::string>();
            auto streams_value = val.as<ov::streams::Num>();
            if (streams_value.num >= 0) {
                streams = streams_value.num;
            } else if (streams_value == ov::streams::NUMA) {
                streams = 1;
            } else if (streams_value == ov::streams::AUTO) {
                streams = ov::threading::IStreamsExecutor::Config::get_default_num_streams();
            } else {
                OPENVINO_THROW("Wrong value for property key ",
                               key,
                               ". Expected non negative numbers (#streams) or ",
                               "ov::streams::NUMA|ov::streams::AUTO, Got: ",
                               value.as<std::string>());
            }
        } else if (ov::inference_num_threads.name() == key) {
            int val;
            try {
                val = value.as<int>();
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value for property key ", key, ". Expected only positive numbers (#threads)");
            }
            if (val < 0) {
                OPENVINO_THROW("Wrong value for property key ", key, ". Expected only positive numbers (#threads)");
            }
            threads = val;
        } else if (ov::internal::threads_per_stream.name() == key) {
            try {
                threads_per_stream = value.as<int>();
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value ", value.as<std::string>(), "for property key ", key);
            }
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
        } else if (ov::hint::num_requests == key) {
            const auto& tmp_val = value.as<std::string>();
            int tmp_i = std::stoi(tmp_val);
            if (tmp_i >= 0)
                num_requests = tmp_i;
            else
                OPENVINO_THROW("Incorrect value, it should be unsigned integer: ", key);
        } else if (ov::log::level == key) {
            log_level = value.as<ov::log::Level>();
        } else if (ov::hint::model_priority == key) {
            model_priority = value.as<ov::hint::Priority>();
        } else if (ov::cache_encryption_callbacks == key) {
            encryption_callbacks = value.as<EncryptionCallbacks>();
        } else if (ov::weights_path == key) {
            weights_path = value.as<std::string>();
            if (!weights_path.empty()) {
                compiled_model_runtime_properties[ov::weights_path.name()] = weights_path.string();
            }
        } else if (ov::cache_mode == key) {
            cache_mode = value.as<CacheMode>();
        } else if (throwOnUnsupported) {
            OPENVINO_THROW("Property was not found: ", key);
        }
    }
}

ov::Any Configuration::Get(const std::string& name) const {
    if (name == ov::device::id) {
        return {std::to_string(device_id)};
    } else if (name == ov::enable_profiling) {
        return {perf_count};
    } else if (name == ov::internal::exclusive_async_requests) {
        return {exclusive_async_requests};
    } else if (name == ov::template_plugin::disable_transformations) {
        return {disable_transformations};
    } else if (name == ov::num_streams) {
        return {std::to_string(streams)};
    } else if (name == ov::inference_num_threads) {
        return {std::to_string(threads)};
    } else if (name == ov::internal::threads_per_stream) {
        return {std::to_string(threads_per_stream)};
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
    } else if (name == ov::weights_path) {
        return weights_path.string();
    } else if (name == ov::internal::compiled_model_runtime_properties) {
        return compiled_model_runtime_properties;
    } else if (name == ov::cache_mode) {
        return cache_mode;
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
}
