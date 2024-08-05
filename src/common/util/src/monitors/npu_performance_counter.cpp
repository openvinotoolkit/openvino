// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/npu_performance_counter.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "openvino/util/monitors/npu_performance_counter.hpp"
#include "openvino/util/monitors/performance_counter.hpp"
#ifdef _WIN32
#    define NOMINMAX
#    include <windows.h>

#    include <string>

#    include "ze_api.h"
#    include "zes_api.h"

namespace ov {
namespace util {
namespace monitor {

class NpuPerformanceCounter::PerformanceCounterImpl {
public:
    bool init_levelZero() {
        ze_result_t result;
        result = zeInit(ZE_INIT_FLAG_VPU_ONLY);
        if (result != ZE_RESULT_SUCCESS) {
            std::cout << "Ze Driver not initialized: " << std::to_string(result) << std::endl;
            return false;
        }
        std::cout << "Ze Driver initialized.\n";
        return true;
    }

    PerformanceCounterImpl() {
        init_levelZero();
    }

    std::map<std::string, int> getDevices() {
        std::map<std::string, int> devices;
        uint32_t driverCount = -1;
        auto status = zeDriverGet(&driverCount, nullptr);
        if (status != ZE_RESULT_SUCCESS) {
            return {};
        }

        std::vector<ze_driver_handle_t> drivers(driverCount);
        status = zeDriverGet(&driverCount, drivers.data());
        if (status != ZE_RESULT_SUCCESS) {
            return {};
        }

        for (const auto driver : drivers) {
            uint32_t deviceCount = 0;
            zeDeviceGet(driver, &deviceCount, nullptr);
            std::vector<ze_device_handle_t> device_handles(deviceCount);
            zeDeviceGet(driver, &deviceCount, device_handles.data());
            std::vector<ze_device_handle_t> found_devices;
            // for each device, find the first one matching the type
            for (uint32_t deviceId = 0; deviceId < deviceCount; ++deviceId) {
                auto phDevice = device_handles[deviceId];
                ze_device_properties_t device_properties = {};
                device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                zeDeviceGetProperties(phDevice, &device_properties);
                found_devices.push_back(phDevice);
                ze_driver_properties_t driver_properties = {};
                driver_properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
                zeDriverGetProperties(driver, &driver_properties);
                std::cout << "Found device: " << device_properties.name << std::endl;
                std::cout << "Driver version: " << driver_properties.driverVersion << "\n";
                ze_api_version_t version = {};
                zeDriverGetApiVersion(driver, &version);
                std::cout << "API version: " << std::to_string(version) << "\n";
            }
            std::move(found_devices.begin(), found_devices.end(), std::back_inserter(device_handle_list));
        }

        return devices;
    }

    std::map<std::string, double> get_load() {
        std::map<std::string, double> devices_load;
        auto devices = getDevices();
        for (int device_id = 0; device_id < device_handle_list.size(); device_id++) {
            devices_load[std::to_string(device_id)] = 0.0;
            const auto& device_handle = device_handle_list[device_id];
            uint32_t engineGroupCount = 0;
            auto status = zesDeviceEnumEngineGroups(device_handle, &engineGroupCount, nullptr);
            std::vector<zes_engine_handle_t> engine_handles(engineGroupCount);
            status = zesDeviceEnumEngineGroups(device_handle, &engineGroupCount, engine_handles.data());
            if (status != ZE_RESULT_SUCCESS) {
                std::cout << "No Engine Modules Found. Status: " << std::to_string(status) << std::endl;
                break;
            }
            std::vector<zes_engine_stats_t> engine_starts(engineGroupCount);
            std::vector<zes_engine_stats_t> engine_ends(engineGroupCount);

            for (int engine_id = 0; engine_id < engine_handles.size(); engine_id++) {
                const auto engine_handle = engine_handles[engine_id];
                auto status = zesEngineGetActivity(engine_handle, &engine_starts[engine_id]);
                if (status != ZE_RESULT_SUCCESS) {
                    std::cout << "Failed to get Engine Stats. Status: " << std::to_string(status) << std::endl;
                    continue;
                }
            }
            auto ts = std::chrono::system_clock::now();
            if (ts > lastTimeStamp) {
                auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
                                                                                   lastTimeStamp);
                if (delta.count() < 500) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500 - delta.count()));
                }
            }
            lastTimeStamp = std::chrono::system_clock::now();

            double utilization = 0.0;
            for (int engine_id = 0; engine_id < engine_handles.size(); engine_id++) {
                const auto engine_handle = engine_handles[engine_id];
                status = zesEngineGetActivity(engine_handle, &engine_ends[engine_id]);
                if (status != ZE_RESULT_SUCCESS) {
                    std::cout << "Failed to zesEngineGetActivity() " << std::endl;
                    break;
                }
                if (engine_starts[engine_id].timestamp != 0) {
                    float engine_utilization = (static_cast<double>(engine_ends[engine_id].activeTime) -
                                                static_cast<double>(engine_starts[engine_id].activeTime)) /
                                               (static_cast<double>(engine_ends[engine_id].timestamp) -
                                                static_cast<double>(engine_starts[engine_id].timestamp));
                    std::cout << "\tDevice: " << device_id << "\t engine: " << engine_id
                              << "\tutilization: " << engine_utilization << std::endl;
                    utilization += engine_utilization;
                }
            }
            std::cout << "Device: " << device_id << "\tutilization: " << utilization << std::endl;
            devices_load[std::to_string(device_id)] = utilization;
        }
        return devices_load;
    }

private:
    std::size_t numDevices = 0;
    std::vector<ze_device_handle_t> device_handle_list;
    std::chrono::time_point<std::chrono::system_clock> lastTimeStamp = std::chrono::system_clock::now();
};

#elif defined(__linux__)
#    include <unistd.h>

#    include <chrono>
#    include <fstream>
#    include <regex>
#    include <utility>

namespace ov {
namespace util {
namespace monitor {
class NpuPerformanceCounter::PerformanceCounterImpl {
public:
    PerformanceCounterImpl() {}

    std::map<std::string, double> get_load() {
        // TODO: Implement.
        return {{"00000000", 0}};
    }
};

#else
namespace ov {
namespace util {
namespace monitor {
// not implemented
class GpuPerformanceCounter::PerformanceCounterImpl {
public:
    std::map<std::string, double> get_load() {
        return {{"00000000", 0}};
    }
};
#endif
NpuPerformanceCounter::NpuPerformanceCounter() : PerformanceCounter("NPU") {}
std::map<std::string, double> NpuPerformanceCounter::get_load() {
    if (!performance_counter)
        performance_counter = std::make_shared<PerformanceCounterImpl>();
    return performance_counter->get_load();
}
}
}
}