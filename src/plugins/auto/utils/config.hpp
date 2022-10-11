// Copyright (C) 2018-2022 Intel Corporation
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
                _modelPriority(0),
                _deviceBindBuffer(false),
                _logLevel("LOG_NONE") {
        adjustKeyMapValues();
    }

    void UpdateFromMap(const std::map<std::string, std::string>& config, const std::string& pluginName) {
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
                if (kvp.second == PluginConfigParams::NO) _disableAutoBatching = true;
                else if (kvp.second == PluginConfigParams::YES) _disableAutoBatching = false;
                else
                    IE_THROW() << "Unsupported config value: " << kvp.second
                            << " for key: " << kvp.first;
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
            } else if (_availableDevices.end() !=
                   std::find(_availableDevices.begin(), _availableDevices.end(), kvp.first)) {
                _passThroughConfig.emplace(kvp.first, kvp.second);
            } else if (kvp.first.find("AUTO_") == 0) {
                _passThroughConfig.emplace(kvp.first, kvp.second);
            } else {
                if (pluginName.find("AUTO") != std::string::npos)
                    IE_THROW(NotFound) << "Unsupported property " << kvp.first;
                else
                    _passThroughConfig.emplace(kvp.first, kvp.second);
            }
        }
        if (!config.empty())
            _keyConfigMap.clear();
        adjustKeyMapValues();
    }
    std::vector<std::string> ParsePrioritiesDevices(const std::string& priorities, const char separator = ',') const {
        std::vector<std::string> devices;
        std::string::size_type pos = 0;
        std::string::size_type endpos = 0;
        auto isAvailableDevice = [&](std::string& deviceName) -> bool {
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
        };
        while ((endpos = priorities.find(separator, pos)) != std::string::npos) {
            auto subStr = priorities.substr(pos, endpos - pos);
            if (!isAvailableDevice(subStr)) {
                IE_THROW() << "Unavailable device name: " << subStr;
            }
            devices.push_back(subStr);
            pos = endpos + 1;
        }
        auto subStr = priorities.substr(pos, priorities.length() - pos);
        if (!isAvailableDevice(subStr)) {
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

        _keyConfigMap[ov::auto_batch_timeout.name()] = _batchTimeout;

        _keyConfigMap[ov::log::level.name()] = _logLevel;

        // for 2nd properties or independent configs from multi
        for (auto && kvp : _passThroughConfig) {
            _keyConfigMap[kvp.first] = kvp.second;
        }
    }
    bool _useProfiling;
    bool _exclusiveAsyncRequests;
    bool _disableAutoBatching;
    std::string _batchTimeout;
    std::string _devicePriority;
    int _modelPriority;
    bool _deviceBindBuffer;
    std::string _logLevel;
    PerfHintsConfig  _perfHintsConfig;
    std::map<std::string, std::string> _passThroughConfig;
    std::map<std::string, std::string> _keyConfigMap;
    const std::set<std::string> _availableDevices = {"AUTO",
                                                     "CPU",
                                                     "GPU",
                                                     "GNA",
                                                     "TEMPLATE",
                                                     "MYRIAD",
                                                     "HDDL",
                                                     "VPUX",
                                                     "MULTI",
                                                     "HETERO",
                                                     "CUDA",
                                                     "NVIDIA",
                                                     "HPU_GOYA",
                                                     "mock"};
};
} // namespace MultiDevicePlugin