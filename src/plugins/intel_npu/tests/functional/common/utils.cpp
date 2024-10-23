// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <string>

#include "intel_npu/npu_private_properties.hpp"

std::string getBackendName(const ov::Core& core) {
    return core.get_property("NPU", ov::intel_npu::backend_name.name()).as<std::string>();
}

std::vector<std::string> getAvailableDevices(const ov::Core& core) {
    return core.get_property("NPU", ov::available_devices.name()).as<std::vector<std::string>>();
}

std::string modelPriorityToString(const ov::hint::Priority priority) {
    std::ostringstream stringStream;

    stringStream << priority;

    return stringStream.str();
}

std::string removeDeviceNameOnlyID(const std::string& device_name_id) {
    std::string::const_iterator first_digit = device_name_id.cend();
    std::string::const_iterator last_digit = device_name_id.cend();
    for (auto&& it = device_name_id.cbegin(); it != device_name_id.cend(); ++it) {
        if (*it >= '0' && *it <= '9') {
            if (first_digit == device_name_id.cend()) {
                first_digit = it;
            }
            last_digit = it;
        }
    }
    if (first_digit == device_name_id.cend()) {
        return std::string("");
    }
    return std::string(first_digit, last_digit + 1);
}

std::vector<ov::AnyMap> getRWMandatoryPropertiesValues(std::vector<ov::AnyMap> props) {
    ov::hint::PerformanceMode performanceModes[] = {ov::hint::PerformanceMode::LATENCY,
                                                    ov::hint::PerformanceMode::THROUGHPUT,
                                                    ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT};
    for (auto& performanceMode : performanceModes) {
        if (std::find(props.begin(),
                      props.end(),
                      ov::AnyMap{{ov::hint::performance_mode.name(), ov::Any(performanceMode)}}) != props.end()) {
            continue;
        }
        props.push_back({{ov::hint::performance_mode(performanceMode)}});
    }

    if (std::find(props.begin(), props.end(), ov::AnyMap{{ov::hint::num_requests.name(), ov::Any(1)}}) == props.end()) {
        props.push_back({{ov::hint::num_requests(1)}});
    }

    ov::hint::ExecutionMode executionModes[] = {ov::hint::ExecutionMode::PERFORMANCE,
                                                ov::hint::ExecutionMode::ACCURACY};
    for (auto& executionMode : executionModes) {
        if (std::find(props.begin(),
                      props.end(),
                      ov::AnyMap{{ov::hint::execution_mode.name(), ov::Any(executionMode)}}) != props.end()) {
            continue;
        }
        props.push_back({{ov::hint::execution_mode(executionMode)}});
    }

    bool enableProfilingValues[] = {true, false};
    for (auto enableProfilingValue : enableProfilingValues) {
        if (std::find(props.begin(),
                      props.end(),
                      ov::AnyMap{{ov::enable_profiling.name(), ov::Any(enableProfilingValue)}}) != props.end()) {
            continue;
        }
        props.push_back({{ov::enable_profiling(enableProfilingValue)}});
    }

    ov::log::Level logLevels[] = {ov::log::Level::NO,
                                  ov::log::Level::ERR,
                                  ov::log::Level::WARNING,
                                  ov::log::Level::INFO,
                                  ov::log::Level::DEBUG,
                                  ov::log::Level::TRACE};
    for (auto& logLevel : logLevels) {
        if (std::find(props.begin(), props.end(), ov::AnyMap{{ov::log::level.name(), ov::Any(logLevel)}}) !=
            props.end()) {
            continue;
        }
        props.push_back({ov::log::level(logLevel)});
    }

    if (std::find(props.begin(), props.end(), ov::AnyMap{{ov::streams::num.name(), ov::Any(ov::streams::num(3))}}) !=
        props.end()) {
        props.push_back({ov::streams::num(3)});
    }
    return props;
}
