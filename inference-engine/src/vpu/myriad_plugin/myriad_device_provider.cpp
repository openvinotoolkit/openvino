// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_device_provider.h"

#include <vpu/utils/profiling.hpp>

#include <mutex>
#include <algorithm>

#ifndef _WIN32
# include <libgen.h>
# include <dlfcn.h>
#endif

using namespace vpu::MyriadPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::VPUConfigParams;
using namespace std;
using namespace vpu;

static std::mutex device_mutex;

MyriadDeviceProvider::MyriadDeviceProvider(bool forceReset,
    std::shared_ptr<IMvnc> mvnc, const Logger::Ptr& log) :
    _log(log), _mvnc(mvnc) {
    VPU_PROFILE(MyriadDeviceProvider);
    int ncResetAll = forceReset;
    auto status = ncGlobalSetOption(NC_RW_RESET_ALL, &ncResetAll, sizeof(ncResetAll));
    if (status != NC_OK) {
        _log->warning(
            "Failed to set NC_RW_RESET_ALL flag to %d: %s\n",
            ncResetAll, Mvnc::ncStatusToStr(nullptr, status));
    }

    int ncLogLevel = NC_LOG_FATAL;
    switch (log->level()) {
        case LogLevel::Warning:
            ncLogLevel = NC_LOG_WARN;
            break;
        case LogLevel::Info:
            ncLogLevel = NC_LOG_INFO;
            break;
        case LogLevel::Debug:
        case LogLevel::Trace:
            ncLogLevel = NC_LOG_DEBUG;
            break;
        default:
            ncLogLevel = NC_LOG_ERROR;
            break;
    }
    status = ncGlobalSetOption(NC_RW_LOG_LEVEL, &ncLogLevel, sizeof(ncLogLevel));
    if (status != NC_OK) {
        _log->warning(
            "Failed to set NC_RW_LOG_LEVEL flag to %d: %s\n",
            ncLogLevel, Mvnc::ncStatusToStr(nullptr, status));
    }
}

DevicePtr MyriadDeviceProvider::openDevice(std::vector<DevicePtr>& devicePool,
                                     const MyriadConfig& config) {
    VPU_PROFILE(openDevice);
    std::lock_guard<std::mutex> lock(device_mutex);

    auto firstBootedButEmptyDevice = std::find_if(devicePool.begin(), devicePool.end(),
                                                  [&config](const DevicePtr &device) {
                                                      return device->isBooted() && device->isEmpty()
                                                             && device->isSuitableForConfig(config);
                                                  });

    if (firstBootedButEmptyDevice != devicePool.end()) {
        auto &device = *firstBootedButEmptyDevice;
        device->_graphNum = 1;
        return device;
    }

    if (!config.deviceName().empty()) {
        auto firstBootedBySpecificName = std::find_if(devicePool.begin(), devicePool.end(),
                                                      [&](const DevicePtr& device) {
                                                          return device->isBooted() && device->isSuitableForConfig(config);
                                                      });

        if (firstBootedBySpecificName != devicePool.end()) {
            DevicePtr device = *firstBootedBySpecificName;
            if (device->isNotFull()) {
                device->_graphNum++;
                return device;
            } else {
                THROW_IE_EXCEPTION << "Maximum number of networks reached for device: " << config.deviceName();
            }
        }
    }

    ncStatus_t booted = bootNextDevice(devicePool, config);

    // TODO Is any tests for this case? #-19309
    // In case, then there is no another not booted device, use already booted with minimum number of executors
    if (booted != NC_OK) {
        std::vector<DevicePtr> availableDevices;

        // Get all suitable devices
        std::copy_if(devicePool.begin(), devicePool.end(), std::back_inserter(availableDevices),
                     [&config](const DevicePtr &device) {
                         return device->isBooted() && device->isNotFull()
                                && device->isSuitableForConfig(config);
                     });

        // Return mock device. If try infer with it, exception will be thrown
        if (availableDevices.empty() && config.platform() != NC_ANY_PLATFORM) {
            DeviceDesc device;
            device._platform = config.platform();
            device._protocol = config.protocol();
            return std::make_shared<DeviceDesc>(device);
        } else if (availableDevices.empty()) {
            THROW_IE_EXCEPTION << "Can not init Myriad device: " << Mvnc::ncStatusToStr(nullptr, booted);
        }

        auto deviceWithMinExecutors = std::min_element(availableDevices.begin(), availableDevices.end(),
                                                       [](const DevicePtr &lhs, const DevicePtr &rhs) { return lhs->_graphNum < rhs->_graphNum; });

        auto &device = *deviceWithMinExecutors;
        device->_graphNum++;
        return device;
    }

    _log->info("Device #%d %s (%s protocol) allocated", devicePool.size() - 1,
               devicePool.back()->_platform == NC_MYRIAD_X ? "MYRIAD-X" : "MYRIAD-2",
               devicePool.back()->_protocol == NC_USB? "USB" : "PCIe");

    return devicePool.back();
}


void MyriadDeviceProvider::closeDevices(std::vector<DevicePtr> &devicePool, std::shared_ptr<IMvnc> mvnc) {
    VPU_PROFILE(closeDevices);
    std::lock_guard<std::mutex> lock(device_mutex);
    for (auto &device : devicePool) {
        if (device->_deviceHandle != nullptr) {
            auto res = ncDeviceClose(&(device->_deviceHandle), mvnc->watchdogHndl());
            if (res != NC_OK)
                printf("ncDeviceClose failed (%d)\n", static_cast<int>(res));
            device->_deviceHandle = nullptr;
        }
    }
}

ncStatus_t MyriadDeviceProvider::bootNextDevice(std::vector<DevicePtr> &devicePool,
                                          const MyriadConfig& config) {
    VPU_PROFILE(bootNextDevice);
// #-17972, #-16790
#if defined(NO_BOOT)
    if (!devicePool.empty()) {
        _log->info("NO_BOOT support only one device");
        return NC_DEVICE_NOT_FOUND;
    }
#endif

    const ncDevicePlatform_t& configPlatform = config.platform();
    const ncDeviceProtocol_t& configProtocol = config.protocol();
    const std::string& configDevName = config.deviceName();
    PowerConfig powerConfig = config.powerConfig();
    int lastDeviceIdx = devicePool.empty() ? -1 : devicePool.back()->_deviceIdx;

    ncStatus_t statusOpen = NC_ERROR;

    DeviceDesc device;

    std::string dirName;

#if !defined(_WIN32)
    Dl_info info;
    dladdr(&device_mutex, &info);

    if (info.dli_fname != nullptr) {
        std::string dli_fname {info.dli_fname};
        dirName = dirname(&dli_fname[0]);
    }
#endif

    ncDeviceDescr_t in_deviceDesc = {};
    in_deviceDesc.platform = configPlatform;
    in_deviceDesc.protocol = configProtocol;

    if (!configDevName.empty()) {
        auto availableDevicesDesc = _mvnc->AvailableDevicesDesc();
        auto it = std::find_if(availableDevicesDesc.begin(), availableDevicesDesc.end(),
                               [&](const ncDeviceDescr_t& deviceDesc) {
                                   return strncmp(deviceDesc.name, configDevName.c_str(), NC_MAX_NAME_SIZE) == 0;
                               });

        if (it == availableDevicesDesc.end()) {
            THROW_IE_EXCEPTION << "Myriad device: " << configDevName << " not found.";
        } else {
            ncDeviceDescr_t deviceDesc = *it;
            if (configPlatform != NC_ANY_PLATFORM &&
                configPlatform != deviceDesc.platform) {
                THROW_IE_EXCEPTION << "Input value of device name and platform are contradict each other. Device name: " << configDevName
                                   << "Platform: " << configPlatform;
            }
        }

        configDevName.copy(in_deviceDesc.name, NC_MAX_NAME_SIZE - 1);
    }

    statusOpen = ncSetDeviceConnectTimeout(config.deviceConnectTimeout().count());
    if (statusOpen) {
        return statusOpen;
    }

    ncDeviceOpenParams_t deviceOpenParams = {};
    deviceOpenParams.watchdogHndl = _mvnc->watchdogHndl();
    deviceOpenParams.watchdogInterval = config.watchdogInterval().count();
    deviceOpenParams.customFirmwareDirectory = dirName.c_str();

    // Open new device with specific path to FW folder
    statusOpen = ncDeviceOpen(&device._deviceHandle,
                              in_deviceDesc, deviceOpenParams);

    if (statusOpen != NC_OK) {
        ncDeviceClose(&device._deviceHandle, _mvnc->watchdogHndl());
        return statusOpen;
    }

    unsigned int dataLength = sizeof(int);

    ncStatus_t status;

    // Get device protocol
    status = ncDeviceGetOption(device._deviceHandle, NC_RO_DEVICE_PLATFORM,
                               reinterpret_cast<void*>(&device._platform), &dataLength);
    if (status != NC_OK || dataLength != sizeof(device._platform)) {
        _log->warning("Failed to get device platform");
        ncDeviceClose(&device._deviceHandle, _mvnc->watchdogHndl());
        return status != NC_OK ? status : NC_ERROR;     // for dataLength error
    }

    //  Get device platform
    status = ncDeviceGetOption(device._deviceHandle, NC_RO_DEVICE_PROTOCOL,
                               reinterpret_cast<void*>(&device._protocol), &dataLength);
    if (status != NC_OK || dataLength != sizeof(device._protocol)) {
        _log->warning("Failed to get device protocol");
        ncDeviceClose(&device._deviceHandle, _mvnc->watchdogHndl());
        return status != NC_OK ? status : NC_ERROR;     // for dataLength error
    }


    // Get device max executors
    status = ncDeviceGetOption(device._deviceHandle, NC_RO_DEVICE_MAX_GRAPH_NUM,
                               reinterpret_cast<void*>(&device._maxGraphNum), &dataLength);
    if (status != NC_OK || dataLength != sizeof(device._maxGraphNum)) {
        _log->warning("Failed to get maximum supported number of graphs");
        ncDeviceClose(&device._deviceHandle, _mvnc->watchdogHndl());
        return status != NC_OK ? status : NC_ERROR;     // for dataLength error
    }

    // Get device name
    char deviceName[NC_MAX_NAME_SIZE];
    dataLength = NC_MAX_NAME_SIZE;
    status = ncDeviceGetOption(device._deviceHandle, NC_RO_DEVICE_NAME,
                               reinterpret_cast<void*>(&deviceName), &dataLength);
    if (status != NC_OK || dataLength > NC_MAX_NAME_SIZE) {
        _log->warning("Failed to get name of booted device");
        ncDeviceClose(&device._deviceHandle, _mvnc->watchdogHndl());
        return status != NC_OK ? status : NC_ERROR;     // for dataLength error
    } else {
        device._name = deviceName;
    }

    status = ncDeviceSetOption(device._deviceHandle, NC_RW_DEVICE_POWER_CONFIG, reinterpret_cast<void*>(&powerConfig), sizeof(dataLength));

    if (status != NC_OK) {
        _log->warning("Failed to set configuration for Power Manager");
        ncDeviceClose(&device._deviceHandle, _mvnc->watchdogHndl());
        return status;
    }

    /* TODO: what should we do if we do not know maximum available graphs? What if we got number <= 0? */
    device._graphNum = 1;
    device._deviceIdx = lastDeviceIdx + 1;
    devicePool.push_back(std::make_shared<DeviceDesc>(device));
    return NC_OK;
}
