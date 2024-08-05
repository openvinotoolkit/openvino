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
private:
    class Engine {
    public:
        ze_result_t status;
        zes_engine_group_t type;
        zes_engine_handle_t engine_handle;
        zes_engine_stats_t engine_statsT0;
        zes_engine_stats_t engine_statsT1;
        Engine(zes_engine_handle_t engine_handle) {
            this->engine_handle = engine_handle;
            zes_engine_properties_t props;
            status = zesEngineGetProperties(engine_handle, &props);
            type = props.type;
        }
        std::string get_engine_type() {
            return std::to_string(type);
        }
        double get_MemoryUtilizationByNPU() {
            double utilization = 0.0;
            status = zesEngineGetActivity(engine_handle, &engine_statsT1);
            if (status != ZE_RESULT_SUCCESS) {
                std::cout << "Could not get Engine Stats -->" << std::to_string(status) << std::endl;
                return -1.0;
            }
            if (engine_statsT0.timestamp != 0) {
                utilization =
                    (static_cast<double>(engine_statsT1.activeTime) - static_cast<double>(engine_statsT0.activeTime)) /
                    (static_cast<double>(engine_statsT1.timestamp) - static_cast<double>(engine_statsT0.timestamp));
            }
            engine_statsT0 = engine_statsT1;
            return utilization;
        }
    };

    class Device {
    public:
        std::string device_name;
        ze_device_handle_t device_handle;
        ze_device_type_t type;
        std::vector<Engine> engines;
        int id = 0;
        Device(std::string device_name, ze_device_type_t type, ze_device_handle_t device_handle) {
            this->device_handle = device_handle;
            this->device_name = device_name;
            this->type = type;
        }
        void device_init() {
            // Init Engines
            ze_result_t status;
            uint32_t engineGroupCount = 0;
            status = zesDeviceEnumEngineGroups(device_handle, &engineGroupCount, nullptr);
            engines.reserve(engineGroupCount);
            std::vector<zes_engine_handle_t> engine_handles(engineGroupCount);
            status = zesDeviceEnumEngineGroups(device_handle, &engineGroupCount, engine_handles.data());
            if (status != ZE_RESULT_SUCCESS) {
                std::cout << "No Engine Modules Found --> " << std::to_string(status) << std::endl;
                exit(1);
            } else {
                std::cout << "Engine Modules Found --> " << engineGroupCount << std::endl;
            }
            for (const auto engine_handle : engine_handles) {
                Engine e(engine_handle);
                engines.push_back(e);
            }
        }
        std::string get_device_name() {
            return device_name;
        }
    };

    class LevelZero {
    private:
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
        std::vector<ze_device_handle_t> findAllDevices(ze_driver_handle_t pDriver) {
            // get all devices
            uint32_t deviceCount = 0;
            zeDeviceGet(pDriver, &deviceCount, nullptr);
            std::vector<ze_device_handle_t> devices(deviceCount);
            zeDeviceGet(pDriver, &deviceCount, devices.data());
            std::vector<ze_device_handle_t> found;
            // for each device, find the first one matching the type
            for (uint32_t device = 0; device < deviceCount; ++device) {
                auto phDevice = devices[device];
                ze_device_properties_t device_properties = {};
                device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                zeDeviceGetProperties(phDevice, &device_properties);
                if (1) {
                    found.push_back(phDevice);
                    ze_driver_properties_t driver_properties = {};
                    driver_properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
                    zeDriverGetProperties(pDriver, &driver_properties);
                    std::cout << "Found " << std::to_string(device_properties.type) << " device..." << "\n";
                    std::cout << "Driver version: " << driver_properties.driverVersion << "\n";
                    ze_api_version_t version = {};
                    zeDriverGetApiVersion(pDriver, &version);
                    std::cout << "API version: " << std::to_string(version) << "\n";
                }
            }
            return found;
        }

    public:
        std::vector<Device> devices;
        bool flag = 0;
        int selection = 0;
        uint32_t driverCount = 0;
        ze_result_t status;
        ze_driver_handle_t pDriver = nullptr;
        std::vector<ze_device_handle_t> device_handle_list;
        ze_device_properties_t pProperties;
        int init() {
            auto zes_enable_sysman_test = _putenv_s("ZES_ENABLE_SYSMAN", "1");
            auto zes_enable_sysman = std::getenv("ZES_ENABLE_SYSMAN");
            if (zes_enable_sysman == nullptr || std::string("1") != zes_enable_sysman) {
                std::cout << "Warning: environment variable ZES_ENABLE_SYSMAN is not 1. L0 sysman tests may fail."
                          << std::endl;
            }
            flag = init_levelZero();
            if (flag == false) {
                return -1;
            }
            // Get Device_list
            status = zeDriverGet(&driverCount, nullptr);
            if (status != ZE_RESULT_SUCCESS) {
                return -2;
            }
            std::vector<ze_driver_handle_t> drivers(driverCount);
            status = zeDriverGet(&driverCount, drivers.data());
            if (status != ZE_RESULT_SUCCESS) {
                return -3;
            }
            for (const auto driver : drivers) {
                auto found_devices = findAllDevices(driver);
                std::move(found_devices.begin(), found_devices.end(), std::back_inserter(device_handle_list));
            }
            // initiate device objects with the device_handles
            for (int i = 0; i < device_handle_list.size(); i++) {
                pProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                status = zeDeviceGetProperties(device_handle_list[i], &pProperties);
                if (status != ZE_RESULT_SUCCESS) {
                    return -4;
                }
                Device d(pProperties.name, pProperties.type, device_handle_list[i]);
                std::cout << "Added Device : " << pProperties.name << std::endl;
                d.device_init();
                devices.push_back(d);
            }
            return 0;
        }
        int get_number_of_devices() {
            return devices.size();
        }
        std::string get_device_name(int id) {
            return devices[id].get_device_name();
        }
        double get_engine_utilization(int did, int eid) {
            return devices[did].engines[eid].get_MemoryUtilizationByNPU();
        }
        int get_number_of_engines(int id) {
            return devices[id].engines.size();
        }
    };

public:
    PerformanceCounterImpl() {
        zero.init();
    }

    std::map<std::string, int> getDevices() {
        std::map<std::string, int> devices;
        auto deviceSize = zero.get_number_of_devices();
        for (int id = 0; id < deviceSize; id++) {
            devices[std::to_string(id)] = id;
        }
        return devices;
    }

    std::map<std::string, double> get_load() {
        std::map<std::string, double> gpuLoad;
        auto devices = getDevices();
        for (auto item : devices) {
            double id = item.second;
            double value = zero.get_engine_utilization(id, 0);
        }
        auto ts = std::chrono::system_clock::now();
        if (ts > lastTimeStamp) {
            auto delta =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastTimeStamp);
            if (delta.count() < 500) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500 - delta.count()));
            }
        }
        lastTimeStamp = std::chrono::system_clock::now();
        for (auto item : devices) {
            double id = item.second;
            double value = zero.get_engine_utilization(id, 0);
            gpuLoad[item.first] = value;
        }
        return gpuLoad;
    }

private:
    LevelZero zero;
    std::size_t numDevices = 0;
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