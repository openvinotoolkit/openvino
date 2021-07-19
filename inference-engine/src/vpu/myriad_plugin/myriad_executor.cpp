// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <vector>
#include <mutex>
#include <map>
#include <algorithm>
#include <utility>
#include <chrono>
#include <memory>

#include <mvnc.h>
#include <ie_common.h>
#include <thread>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/profiling.hpp>

#include <vpu/configuration/options/protocol.hpp>
#include <vpu/configuration/options/power_config.hpp>
#include <vpu/configuration/options/watchdog_interval.hpp>
#include <vpu/configuration/options/device_id.hpp>
#include <vpu/configuration/options/device_connect_timeout.hpp>
#include <vpu/configuration/options/memory_type.hpp>
#include <vpu/configuration/options/enable_async_dma.hpp>

#include "myriad_executor.h"

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

MyriadExecutor::MyriadExecutor(bool forceReset, std::shared_ptr<IMvnc> mvnc,
    const LogLevel& vpuLogLevel, const Logger::Ptr& log) : _log(log), _mvnc(std::move(mvnc)) {
    VPU_PROFILE(MyriadExecutor);
    VPU_THROW_UNLESS(_mvnc, "mvnc is null");
    int ncResetAll = forceReset;
    auto status = ncGlobalSetOption(NC_RW_RESET_ALL, &ncResetAll, sizeof(ncResetAll));
    if (status != NC_OK) {
        _log->warning(
            "Failed to set NC_RW_RESET_ALL flag to %d: %s\n",
            ncResetAll, ncStatusToStr(nullptr, status));
    }

    int ncLogLevel = NC_LOG_FATAL;
    switch (vpuLogLevel) {
    case LogLevel::Warning:
        ncLogLevel = NC_LOG_WARN;
        break;
    case LogLevel::Info:
        ncLogLevel = NC_LOG_INFO;
        break;
    case LogLevel::Debug:
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
            ncLogLevel, ncStatusToStr(nullptr, status));
    }
}

/*
 * @brief Boot available device
 */
ncStatus_t MyriadExecutor::bootNextDevice(std::vector<DevicePtr> &devicePool, const PluginConfiguration& config) {
    VPU_PROFILE(bootNextDevice);
// #-17972, #-16790
#if defined(NO_BOOT)
    if (!devicePool.empty()) {
        _log->info("NO_BOOT support only one device");
        return NC_DEVICE_NOT_FOUND;
    }
#endif

    const ncDevicePlatform_t& configPlatform = ncDevicePlatform_t::NC_ANY_PLATFORM;
    const ncDeviceProtocol_t& configProtocol = config.get<ProtocolOption>();
    const std::string& configDevName = config.get<DeviceIDOption>();
    PowerConfig powerConfig = config.get<PowerConfigOption>();
    int enableAsyncDma = config.get<EnableAsyncDMAOption>();
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
            IE_THROW() << "Myriad device: " << configDevName << " not found.";
        } else {
            ncDeviceDescr_t deviceDesc = *it;
            if (configPlatform != NC_ANY_PLATFORM &&
                configPlatform != deviceDesc.platform) {
                IE_THROW() << "Input value of device name and platform are contradict each other. Device name: " << configDevName
                                   << "Platform: " << configPlatform;
            }
        }

        configDevName.copy(in_deviceDesc.name, NC_MAX_NAME_SIZE - 1);
    }

    statusOpen = ncSetDeviceConnectTimeout(static_cast<int>(config.get<DeviceConnectTimeoutOption>().count()));
    if (statusOpen) {
        return statusOpen;
    }

    ncDeviceOpenParams_t deviceOpenParams = {};
    deviceOpenParams.watchdogHndl = _mvnc->watchdogHndl();
    deviceOpenParams.watchdogInterval = static_cast<int>(config.get<WatchdogIntervalOption>().count());
    deviceOpenParams.memoryType = static_cast<char>(config.get<MemoryTypeOption>());
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

    status = ncDeviceSetOption(device._deviceHandle, NC_RW_ENABLE_ASYNC_DMA, reinterpret_cast<void*>(&enableAsyncDma), sizeof(dataLength));

    if (status != NC_OK) {
        _log->warning("Failed to set option for async DMA");
        ncDeviceClose(&device._deviceHandle, _mvnc->watchdogHndl());
        return status;
    }

    /* TODO: what should we do if we do not know maximum available graphs? What if we got number <= 0? */
    device._graphNum = 1;
    device._deviceIdx = lastDeviceIdx + 1;
    devicePool.push_back(std::make_shared<DeviceDesc>(device));
    return NC_OK;
}

DevicePtr MyriadExecutor::openDevice(std::vector<DevicePtr>& devicePool,
                                     const PluginConfiguration& config) {
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

    if (!config.get<DeviceIDOption>().empty()) {
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
                IE_THROW() << "Maximum number of networks reached for device: " << config.get<DeviceIDOption>();
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
        if (availableDevices.empty()) {
            IE_THROW() << "Can not init Myriad device: " << ncStatusToStr(nullptr, booted);
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

VPU_PACKED(bin_header {
    int32_t  magic;
    uint32_t frequency;
};)

void MyriadExecutor::closeDevices(std::vector<DevicePtr> &devicePool, std::shared_ptr<IMvnc> mvnc) {
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

void MyriadExecutor::allocateGraph(DevicePtr &device, GraphDesc &graphDesc,
                                   const std::vector<char> &graphFileContent,
                                   const std::pair<const char*, size_t> &graphHeaderDesc,
                                   size_t numStages, const std::string & networkName, int executors) {
    VPU_PROFILE(allocateGraph);
    _numStages = static_cast<int>(numStages);
    graphDesc._name = networkName;
    if (device->_deviceHandle == nullptr) {
        IE_THROW() << "Failed to allocate graph: MYRIAD device is not opened.";
    }

    ncStatus_t status;

    status = ncGraphCreate(networkName.c_str(), &graphDesc._graphHandle);
    if (status != NC_OK) {
        IE_THROW() << "Failed to init graph: " << ncStatusToStr(nullptr, status);
    }

    status = ncGraphSetOption(graphDesc._graphHandle, NC_RW_GRAPH_EXECUTORS_NUM, &executors, sizeof(executors));
    if (status != NC_OK) {
        IE_THROW() << "Failed to set graph executors: " << ncStatusToStr(nullptr, status);
    }

    status = ncGraphAllocate(device->_deviceHandle,
                             graphDesc._graphHandle,
                             graphFileContent.data(),
                             static_cast<unsigned int>(graphFileContent.size()),
                             graphHeaderDesc.first,
                             static_cast<unsigned>(graphHeaderDesc.second));
    if (status != NC_OK) {
        IE_THROW() << "Failed to allocate graph: " << ncStatusToStr(nullptr, status);
    }

    unsigned int dataLength = sizeof(int);

    int numInputs = 0;
    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_INPUT_COUNT, &numInputs, &dataLength);
    if (status != NC_OK) {
        IE_THROW() << "Failed to get number of inputs: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
    if (numInputs != 1) {
        IE_THROW() << "Unsupported number of inputs: " << numInputs;
    }

    int numOutputs = 0;
    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs, &dataLength);
    if (status != NC_OK) {
        IE_THROW() << "Failed to get number of outputs: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
    if (numOutputs != 1) {
        IE_THROW() << "Unsupported number of outputs: " << numOutputs;
    }

    dataLength = sizeof(ncTensorDescriptor_t);
    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &graphDesc._inputDesc,
                              &dataLength);
    if (status != NC_OK) {
        IE_THROW() << "Failed to get input description: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &graphDesc._outputDesc,
                              &dataLength);
    if (status != NC_OK) {
        IE_THROW() << "Failed to get output description: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    unsigned int fifo_elements = (device->_platform == NC_MYRIAD_2 && executors == 1) ? 4 : 2 * executors;

    status = ncFifoCreate("input", NC_FIFO_HOST_WO, &graphDesc._inputFifoHandle);
    if (status != NC_OK) {
        IE_THROW() << "Failed to init input FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoAllocate(graphDesc._inputFifoHandle, device->_deviceHandle, &graphDesc._inputDesc, fifo_elements);
    if (status != NC_OK) {
        IE_THROW() << "Failed to create input FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoCreate("output", NC_FIFO_HOST_RO, &graphDesc._outputFifoHandle);
    if (status != NC_OK) {
        IE_THROW() << "Failed to init output FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoAllocate(graphDesc._outputFifoHandle, device->_deviceHandle, &graphDesc._outputDesc, fifo_elements);
    if (status != NC_OK) {
        IE_THROW() << "Failed to create output FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
}

void MyriadExecutor::queueInference(GraphDesc &graphDesc, void *input_data, size_t input_bytes,
                    void *result_data, size_t result_bytes) {
    VPU_PROFILE(queueInference);
#ifndef NDEBUG
    if (auto dumpFileName = std::getenv("IE_VPU_DUMP_INPUT_FILE_NAME")) {
        std::ofstream file(dumpFileName, std::ios_base::binary | std::ios_base::out);
        if (!file.is_open()) {
            IE_THROW() << "[VPU] Cannot open file " << dumpFileName << " for writing";
        }
        file.write(static_cast<const char*>(input_data), input_bytes);
    }
#endif

    if (graphDesc._inputDesc.totalSize != input_bytes) {
        IE_THROW() << "Input has unexpected size " << input_bytes << ", expected "
                           << graphDesc._inputDesc.totalSize;
    }

    ncStatus_t status = ncGraphQueueInferenceWithFifoElem(graphDesc._graphHandle,
                                graphDesc._inputFifoHandle, graphDesc._outputFifoHandle,
                                input_data, &graphDesc._inputDesc.totalSize, nullptr);
    if (status != NC_OK) {
        IE_THROW() << "Failed to queue inference: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    if (result_data != nullptr && result_bytes != 0) {
        getResult(graphDesc, result_data, static_cast<unsigned>(result_bytes));
    }
}

void MyriadExecutor::getResult(GraphDesc &graphDesc, void *result_data, unsigned int result_bytes) {
    ncStatus_t status;
    void *userParam = nullptr;
    status = ncFifoReadElem(graphDesc._outputFifoHandle, result_data, &result_bytes, &userParam);
    if (status != NC_OK) {
        IE_THROW() << "Failed to read output from FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
}

void MyriadExecutor::deallocateGraph(DevicePtr &device, GraphDesc &graphDesc) {
    VPU_PROFILE(deallocateGraph);
    std::lock_guard<std::mutex> lock(device_mutex);

    if (graphDesc._inputFifoHandle != nullptr) {
        auto res = ncFifoDestroy(&graphDesc._inputFifoHandle);
        if (res != NC_OK)
            _log->warning("ncFifoDelete result %s", ncStatusToStr(nullptr, res));
        graphDesc._inputFifoHandle = nullptr;
    }
    if (graphDesc._outputFifoHandle != nullptr) {
        auto res = ncFifoDestroy(&graphDesc._outputFifoHandle);
        if (res != NC_OK)
            _log->warning("ncFifoDelete result %s", ncStatusToStr(nullptr, res));
        graphDesc._outputFifoHandle = nullptr;
    }
    if (graphDesc._graphHandle != nullptr) {
        auto res = ncGraphDestroy(&graphDesc._graphHandle);
        if (res !=NC_OK)
            _log->warning("Deallocate Graph result %s.", ncStatusToStr(nullptr, res));
        graphDesc._graphHandle = nullptr;
    }
    if (device->_deviceHandle != nullptr) {
        device->_graphNum -= 1;
    }
}

std::string MyriadExecutor::ncStatusToStr(ncGraphHandle_t *graphHandle, ncStatus_t status) {
#define MVNC_STATUS_TO_STR(E) case E: return #E;
    switch (status) {
        MVNC_STATUS_TO_STR(NC_OK)
        MVNC_STATUS_TO_STR(NC_BUSY)
        MVNC_STATUS_TO_STR(NC_ERROR)
        MVNC_STATUS_TO_STR(NC_OUT_OF_MEMORY)
        MVNC_STATUS_TO_STR(NC_DEVICE_NOT_FOUND)
        MVNC_STATUS_TO_STR(NC_INVALID_PARAMETERS)
        MVNC_STATUS_TO_STR(NC_TIMEOUT)
        MVNC_STATUS_TO_STR(NC_MVCMD_NOT_FOUND)
        MVNC_STATUS_TO_STR(NC_NOT_ALLOCATED)
        MVNC_STATUS_TO_STR(NC_UNAUTHORIZED)
        MVNC_STATUS_TO_STR(NC_UNSUPPORTED_FEATURE)
        MVNC_STATUS_TO_STR(NC_UNSUPPORTED_GRAPH_FILE)
        MVNC_STATUS_TO_STR(NC_UNSUPPORTED_CONFIGURATION_FILE)
        case NC_MYRIAD_ERROR: {
            if (graphHandle == nullptr) {
                return "NC_MYRIAD_ERROR";
            } else {
                auto debugInfo = getGraphInfo<char>(graphHandle, NC_RO_GRAPH_DEBUG_INFO, NC_DEBUG_BUFFER_SIZE);
                if (debugInfo.empty()) {
                    return "NC_MYRIAD_ERROR";
                } else {
                    return std::string(debugInfo.begin(), debugInfo.end());
                }
            }
        }
        default:
            return "UNKNOWN MVNC STATUS";
    }
#undef MVNC_STATUS_TO_STR
}

void MyriadExecutor::printThrottlingStatus() {
// TODO: enable when needed
}

float MyriadExecutor::GetThermal(const DevicePtr& device) {
    unsigned int thermal_stats_len = NC_THERMAL_BUFFER_SIZE;
    static_assert(NC_THERMAL_BUFFER_SIZE % sizeof(float) == 0,
                  "NC_THERMAL_BUFFER_SIZE is not divisible by sizeof(float)");
    float thermal_stats[NC_THERMAL_BUFFER_SIZE / sizeof(float)];
    ncStatus_t status = ncDeviceGetOption(device->_deviceHandle,
                                          NC_RO_DEVICE_THERMAL_STATS,
                                          reinterpret_cast<void *>(&thermal_stats),
                                          &thermal_stats_len);

    if (status != NC_OK) {
        IE_THROW() << "Failed to get thermal stats: " << ncStatusToStr(nullptr, status);
    } else {
        return thermal_stats[0];
    }
}

std::vector<float> MyriadExecutor::getPerfTimeInfo(ncGraphHandle_t *graphHandle) {
    return getGraphInfo<float>(graphHandle, NC_RO_GRAPH_TIME_TAKEN, _numStages + 2);
}
