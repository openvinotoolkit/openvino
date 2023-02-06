// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <ie_parameter.hpp>
#include <ie_performance_hints.hpp>
#include "log.hpp"
#include <string>
#include <map>
#include <vector>

namespace MultiDevicePlugin {
using namespace InferenceEngine;

struct PluginConfig {
    PluginConfig():
                _useProfiling(false),
                _exclusiveAsyncRequests(false),
                _disableAutoBatching(false),
                _batchTimeout("1000"),
                _devicePriority(""),
                _modelPriority(1),
                _deviceBindBuffer(false),
                _enableStartupFallback(true),
                _logLevel("LOG_NONE") {
        adjustKeyMapValues();
    }
    std::vector<std::string> supportedConfigKeys(const std::string& pluginName = "AUTO") const {
        std::vector<std::string> supported_configKeys = []() -> decltype(PerfHintsConfig::SupportedKeys()) {
            auto res = PerfHintsConfig::SupportedKeys();
            res.push_back(ov::device::priorities.name());
            res.push_back(ov::enable_profiling.name());
            res.push_back(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS);
            res.push_back(ov::hint::model_priority.name());
            res.push_back(ov::hint::allow_auto_batching.name());
            res.push_back(ov::log::level.name());
            res.push_back(ov::intel_auto::device_bind_buffer.name());
            res.push_back(ov::auto_batch_timeout.name());
            res.push_back(ov::intel_auto::enable_startup_fallback.name());
            return res;
        }();
        auto multi_supported_configKeys = supported_configKeys;
        return pluginName == "AUTO" ? supported_configKeys : multi_supported_configKeys;
    }

    std::vector<ov::PropertyName> supportedProperties(const std::string& pluginName = "AUTO") const {
        std::vector<ov::PropertyName> supported_properties = []() -> std::vector<ov::PropertyName> {
            auto RO_property = [](const std::string& propertyName) {
                return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
            };
            auto RW_property = [](const std::string& propertyName) {
                return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
            };
            std::vector<ov::PropertyName> roProperties{RO_property(ov::supported_properties.name()),
                                                       RO_property(ov::device::full_name.name()),
                                                       RO_property(ov::device::capabilities.name())};
            // the whole config is RW before network is loaded.
            std::vector<ov::PropertyName> rwProperties{RW_property(ov::hint::model_priority.name()),
                                                       RW_property(ov::log::level.name()),
                                                       RW_property(ov::device::priorities.name()),
                                                       RW_property(ov::enable_profiling.name()),
                                                       RW_property(ov::hint::allow_auto_batching.name()),
                                                       RW_property(ov::auto_batch_timeout.name()),
                                                       RW_property(ov::hint::performance_mode.name()),
                                                       RW_property(ov::hint::num_requests.name()),
                                                       RW_property(ov::intel_auto::device_bind_buffer.name()),
                                                       RW_property(ov::cache_dir.name()),
                                                       RW_property(ov::intel_auto::enable_startup_fallback.name())};
            std::vector<ov::PropertyName> supportedProperties;
            supportedProperties.reserve(roProperties.size() + rwProperties.size());
            supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
            supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());
            return supportedProperties;
        }();
        auto multi_supported_properties = supported_properties;
        return pluginName == "AUTO" ? supported_properties : multi_supported_properties;
    }

    std::vector<std::string> supportedMetrics(const std::string& pluginName = "AUTO") const {
        std::vector<std::string> supported_metrics = []() -> std::vector<std::string> {
            std::vector<std::string> metrics;
            metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
            metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
            metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
            metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
            return metrics;
        }();
        auto multi_supported_metrics = supported_metrics;
        return pluginName == "AUTO" ? supported_metrics : multi_supported_metrics;
    }

    void UpdateFromMap(const std::map<std::string, std::string>& config, const std::string& pluginName, bool supportHWProprety = false) {
        const auto perf_hints_configs = PerfHintsConfig::SupportedKeys();
        for (auto&& kvp : config) {
            if (kvp.first == ov::enable_profiling) {
                if (kvp.second == PluginConfigParams::YES) {
                    _useProfiling = true;
                } else if (kvp.second == PluginConfigParams::NO) {
                    _useProfiling = false;
                } else {
                    IE_THROW() << "Unsupported config value: " << kvp.second
                            << " for key: " << kvp.first;
                }
            } else if (kvp.first == PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) {
                if (kvp.second == PluginConfigParams::YES) _exclusiveAsyncRequests = true;
                else if (kvp.second == PluginConfigParams::NO) _exclusiveAsyncRequests = false;
                else
                    IE_THROW() << "Unsupported config value: " << kvp.second
                            << " for key: " << kvp.first;
            } else if (kvp.first == ov::log::level.name()) {
                _logLevel = kvp.second;
                auto success = setLogLevel(_logLevel);
                if (!success) {
                    IE_THROW() << "Unsupported config value: " << kvp.second
                                << " for key: " << kvp.first;
                }
            } else if (kvp.first == ov::hint::model_priority) {
                if (kvp.second == "LOW" ||
                    kvp.second == CONFIG_VALUE(MODEL_PRIORITY_LOW)) {
                    _modelPriority = 2;
                } else if (kvp.second == "MEDIUM" ||
                    kvp.second == CONFIG_VALUE(MODEL_PRIORITY_MED)) {
                    _modelPriority = 1;
                } else if (kvp.second == "HIGH" ||
                    kvp.second == CONFIG_VALUE(MODEL_PRIORITY_HIGH)) {
                    _modelPriority = 0;
                } else {
                    IE_THROW() << "Unsupported config value: " << kvp.second
                        << " for key: " << kvp.first;
                }
            } else if (kvp.first == ov::hint::allow_auto_batching) {
                if (kvp.second == PluginConfigParams::NO) {
                    _disableAutoBatching = true;
                    // temp flag, to be removed when unify this key to ie core
                    _isBatchConfigSet = true;
                } else if (kvp.second == PluginConfigParams::YES) {
                    _disableAutoBatching = false;
                    _isBatchConfigSet = true;
                } else {
                    IE_THROW() << "Unsupported config value: " << kvp.second
                            << " for key: " << kvp.first;
                }
            } else if (kvp.first == ov::auto_batch_timeout) {
                try {
                    auto batchTimeout = std::stoi(kvp.second);
                    if (batchTimeout < 0) {
                        IE_THROW() << "Unsupported config value: " << kvp.second
                            << " for key: " << kvp.first;
                    }
                    _batchTimeout = kvp.second;
                } catch (...) {
                    IE_THROW() << "Unsupported config value: " << kvp.second
                            << " for key: " << kvp.first;
                }
            } else if (kvp.first == ov::intel_auto::device_bind_buffer.name()) {
                if (kvp.second == PluginConfigParams::YES) _deviceBindBuffer = true;
                else if (kvp.second == PluginConfigParams::NO) _deviceBindBuffer = false;
                else
                    IE_THROW() << "Unsupported config value: " << kvp.second
                            << " for key: " << kvp.first;
            } else if (kvp.first == ov::device::priorities.name()) {
                if (!kvp.second.empty())
                    ParsePrioritiesDevices(kvp.second);
                _devicePriority = kvp.second;
            } else if (std::find(perf_hints_configs.begin(), perf_hints_configs.end(), kvp.first) != perf_hints_configs.end()) {
                _perfHintsConfig.SetConfig(kvp.first, kvp.second);
                // if first level property has perf_hint setting
                if (kvp.first == ov::hint::performance_mode.name())
                    _isSetPerHint = true;
            } else if (_availableDevices.end() != std::find(_availableDevices.begin(),
                                                            _availableDevices.end(),
                                                            DeviceIDParser(kvp.first).getDeviceName())) {
                // AUTO and MULTI can accept secondary properites on calling both core::comile_model() and
                // core::set_property().
                _passThroughConfig.emplace(kvp.first, kvp.second);
            } else if (kvp.first.find("AUTO_") == 0) {
                _passThroughConfig.emplace(kvp.first, kvp.second);
            } else if (kvp.first == ov::cache_dir.name()) {
                _cacheDir = kvp.second;
                _isSetCacheDir = true;
            } else if (kvp.first == ov::intel_auto::enable_startup_fallback.name()) {
                if (kvp.second == PluginConfigParams::YES) _enableStartupFallback = true;
                else if (kvp.second == PluginConfigParams::NO) _enableStartupFallback = false;
                else
                    IE_THROW() << "Unsupported config value: " << kvp.second
                            << " for key: " << kvp.first;
            } else {
                if (pluginName.find("AUTO") != std::string::npos || !supportHWProprety)
                    // AUTO and MULTI just only accept its own properites and secondary property when calling
                    // core::set_property().
                    IE_THROW(NotFound) << "Unsupported property " << kvp.first;

                // MULTI could accept the HW primary property like {"NUM_STREAMS", "4"}
                _passThroughConfig.emplace(kvp.first, kvp.second);
            }
        }
        if (!config.empty())
            _keyConfigMap.clear();
        adjustKeyMapValues();
    }
    bool isSupportedDevice(const std::string& deviceName) const {
        if (deviceName.empty())
            return false;
        auto realDevName = deviceName[0] != '-' ? deviceName : deviceName.substr(1);
        if (realDevName.empty()) {
            return false;
        }
        realDevName = DeviceIDParser(realDevName).getDeviceName();
        std::string::size_type realEndPos = 0;
        if ((realEndPos = realDevName.find('(')) != std::string::npos) {
            realDevName = realDevName.substr(0, realEndPos);
        }
        if (_availableDevices.end() == std::find(_availableDevices.begin(), _availableDevices.end(), realDevName)) {
            return false;
        }
        return true;
    }
    std::vector<std::string> ParsePrioritiesDevices(const std::string& priorities, const char separator = ',') const {
        std::vector<std::string> devices;
        std::string::size_type pos = 0;
        std::string::size_type endpos = 0;
        while ((endpos = priorities.find(separator, pos)) != std::string::npos) {
            auto subStr = priorities.substr(pos, endpos - pos);
            if (!isSupportedDevice(subStr)) {
                IE_THROW() << "Unavailable device name: " << subStr;
            }
            devices.push_back(subStr);
            pos = endpos + 1;
        }
        auto subStr = priorities.substr(pos, priorities.length() - pos);
        if (!isSupportedDevice(subStr)) {
            IE_THROW() << "Unavailable device name: " << subStr;
        }
        devices.push_back(subStr);
        return devices;
    }
    void adjustKeyMapValues() {
        if (_useProfiling) {
            _keyConfigMap[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
        } else {
            _keyConfigMap[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::NO;
        }
        if (_exclusiveAsyncRequests)
            _keyConfigMap[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::YES;
        else
            _keyConfigMap[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::NO;
        {
            std::string priority;
            if (_modelPriority == 0)
                priority = ov::util::to_string(ov::hint::Priority::HIGH);
            else if (_modelPriority == 1)
                priority = ov::util::to_string(ov::hint::Priority::MEDIUM);
            else
                priority = ov::util::to_string(ov::hint::Priority::LOW);
            _keyConfigMap[ov::hint::model_priority.name()] = priority;
        }
        _keyConfigMap[PluginConfigParams::KEY_PERFORMANCE_HINT] = _perfHintsConfig.ovPerfHint;
        _keyConfigMap[PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS] = std::to_string(_perfHintsConfig.ovPerfHintNumRequests);

        _keyConfigMap[ov::device::priorities.name()] = _devicePriority;

        if (_disableAutoBatching)
            _keyConfigMap[ov::hint::allow_auto_batching.name()] = PluginConfigParams::NO;
        else
            _keyConfigMap[ov::hint::allow_auto_batching.name()] = PluginConfigParams::YES;
        if (_deviceBindBuffer)
            _keyConfigMap[ov::intel_auto::device_bind_buffer.name()] = PluginConfigParams::YES;
        else
            _keyConfigMap[ov::intel_auto::device_bind_buffer.name()] = PluginConfigParams::NO;
        if (_enableStartupFallback)
            _keyConfigMap[ov::intel_auto::enable_startup_fallback.name()] = PluginConfigParams::YES;
        else
            _keyConfigMap[ov::intel_auto::enable_startup_fallback.name()] = PluginConfigParams::NO;
        _keyConfigMap[ov::auto_batch_timeout.name()] = _batchTimeout;

        _keyConfigMap[ov::log::level.name()] = _logLevel;

        _keyConfigMap[ov::cache_dir.name()] = _cacheDir;

        // for 2nd properties or independent configs from multi
        for (auto && kvp : _passThroughConfig) {
            _keyConfigMap[kvp.first] = kvp.second;
        }
    }
    std::string _cacheDir{};
    bool _useProfiling;
    bool _exclusiveAsyncRequests;
    bool _disableAutoBatching;
    std::string _batchTimeout;
    std::string _devicePriority;
    int _modelPriority;
    bool _deviceBindBuffer;
    bool _enableStartupFallback;
    std::string _logLevel;
    PerfHintsConfig  _perfHintsConfig;
    // Add this flag to check if user app sets hint with none value that is equal to the default value of hint.
    bool _isSetPerHint = false;
    bool _isSetCacheDir = false;
    bool _isBatchConfigSet = false;
    std::map<std::string, std::string> _passThroughConfig;
    std::map<std::string, std::string> _keyConfigMap;
    const std::set<std::string> _availableDevices =
        {"AUTO", "CPU", "GPU", "TEMPLATE", "MYRIAD", "VPUX", "MULTI", "HETERO", "mock"};
};
} // namespace MultiDevicePlugin
