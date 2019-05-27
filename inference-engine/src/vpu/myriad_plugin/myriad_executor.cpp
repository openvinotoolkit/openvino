// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <map>
#include <algorithm>
#include <utility>

#include <mvnc.h>
#include <ie_common.h>
#include <thread>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/logger.hpp>

#include "myriad_executor.h"
#include "myriad_config.h"

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

MyriadExecutor::MyriadExecutor(bool forceReset, const LogLevel& vpuLogLevel, const Logger::Ptr& log) : _log(log) {
    int ncLogLevel;
    switch (vpuLogLevel) {
        case LogLevel::Warning:
            ncLogLevel = 2;
            break;
        case LogLevel::Info:
            ncLogLevel = 1;
            break;
        case LogLevel::Debug:
            ncLogLevel = 0;
            break;
        default:
            ncLogLevel = 3;
            break;
    }

    int reset_all = forceReset;
    char * tmp = std::getenv("VPU_FORCE_RESET");
    if (tmp) {
        std::string env = tmp;
        if (env == "0")
            reset_all = 0;
        else if (env == "1")
            reset_all = 1;
    }

    auto status = ncGlobalSetOption(NC_RW_RESET_ALL, &reset_all, sizeof(reset_all));
    if (status != NC_OK) {
        _log->warning("failed to set RESET_ALL flag: %d with error: %s\n",
                    ncLogLevel,
                    ncStatusToStr(nullptr, status));
    }

    status = ncGlobalSetOption(NC_RW_LOG_LEVEL, &ncLogLevel, sizeof(ncLogLevel));
    if (status != NC_OK) {
        _log->warning("failed to set log level: %d with error: %s\n",
                    ncLogLevel,
                    ncStatusToStr(nullptr, status));
    }
}

/*
 * @brief Boot available device
 */
ncStatus_t MyriadExecutor::bootNextDevice(std::vector<DevicePtr> &devicePool,
                                          const ncDevicePlatform_t &configPlatform,
                                          int watchdogInterval) {
// #-17972, #-16790
#if defined(USE_PCIE) || defined(NO_BOOT)
    if (!devicePool.empty()) {
        _log->info("PCIe and NO_BOOT support only one device");
        return NC_DEVICE_NOT_FOUND;
    }
#endif
    int lastDeviceIdx = devicePool.empty() ? -1 : devicePool.back()->_deviceIdx;

    ncStatus_t statusOpen = NC_ERROR;

    DeviceDesc device;

    char* dirName = nullptr;

#if !defined(_WIN32)
    Dl_info info;
    dladdr(&device_mutex, &info);
    char* dli_fname = nullptr;

    if (info.dli_fname != nullptr) {
        dli_fname = strdup(info.dli_fname);
        dirName = dirname(dli_fname);
    }
#endif

    // Open new device with specific path to FW folder
    statusOpen = ncDeviceOpen(&device._deviceHandle, configPlatform, watchdogInterval, dirName);

#if !defined(_WIN32)
    if (info.dli_fname != nullptr) {
        free(dli_fname);
    }
#endif

    if (statusOpen != NC_OK) {
        ncDeviceClose(&device._deviceHandle);
        return statusOpen;
    }

    unsigned int dataLength;

    // Get device platform
    ncStatus_t status = ncDeviceGetOption(device._deviceHandle, NC_RO_DEVICE_PLATFORM,
                               reinterpret_cast<void*>(&device._platform), &dataLength);
    if (status != NC_OK || dataLength != sizeof(device._platform)) {
        _log->warning("Failed to get device platform");
        ncDeviceClose(&device._deviceHandle);
        return status != NC_OK ? status : NC_ERROR;     // for dataLength error
    }

    // Get device max executors
    status = ncDeviceGetOption(device._deviceHandle, NC_RO_DEVICE_MAX_GRAPH_NUM,
                               reinterpret_cast<void*>(&device._maxExecutors), &dataLength);
    if (status != NC_OK || dataLength != sizeof(device._maxExecutors)) {
        _log->warning("Failed to get maximum supported number of graphs");
        ncDeviceClose(&device._deviceHandle);
        return status != NC_OK ? status : NC_ERROR;     // for dataLength error
    }

    /* TODO: what should we do if we do not know maximum available graphs? What if we got number <= 0? */
    device._executors = 1;
    device._deviceIdx = lastDeviceIdx + 1;
    devicePool.push_back(std::make_shared<DeviceDesc>(device));
    return NC_OK;
}

DevicePtr MyriadExecutor::openDevice(std::vector<DevicePtr> &devicePool,
                                     const std::shared_ptr<MyriadConfig> &config) {
    std::lock_guard<std::mutex> lock(device_mutex);

    auto firstBootedButEmptyDevice = std::find_if(devicePool.begin(), devicePool.end(),
        [&config](const DevicePtr &device) {
            bool isFromConfig = config->platform == UNKNOWN_PLATFORM ? true : device->_platform == config->platform;
            return device->isBooted() && device->isEmpty() && isFromConfig;
        });

    if (firstBootedButEmptyDevice != devicePool.end()) {
        auto &device = *firstBootedButEmptyDevice;
        device->_executors = 1;
        return device;
    }

    ncStatus_t booted = bootNextDevice(devicePool, config->platform, config->watchdogInterval);

    // TODO Is any tests for this case?
    // In case, then there is no another not booted device, use already booted with minimum number of executors
    if (booted != NC_OK) {
        std::vector<DevicePtr> availableDevices;
        // Get all suitable devices
        std::copy_if(devicePool.begin(), devicePool.end(), std::back_inserter(availableDevices),
            [&config](const DevicePtr &device) {
                bool isFromConfig = config->platform == UNKNOWN_PLATFORM ? true : device->_platform == config->platform;
                return !device->isEmpty() && device->isAvailable() && isFromConfig;
            });

        // Return mock device. If try infer with it, exception will be thrown
        if (availableDevices.empty() && config->platform != UNKNOWN_PLATFORM) {
            DeviceDesc device;
            device._platform = config->platform;
            return std::make_shared<DeviceDesc>(device);
        } else if (availableDevices.empty()) {
            THROW_IE_EXCEPTION << "Can not init USB device: " << ncStatusToStr(nullptr, booted);
        }

        auto deviceWithMinExecutors = std::min_element(availableDevices.begin(), availableDevices.end(),
            [](const DevicePtr &lhs, const DevicePtr &rhs) { return lhs->_executors < rhs->_executors; });

        auto &device = *deviceWithMinExecutors;
        device->_executors++;
        return device;
    }

    _log->info("Device #%d %s allocated", devicePool.size() - 1,
        devicePool.back()->_platform == MYRIAD_X ? "MYRIAD-X" : "MYRIAD-2");

    return devicePool.back();
}

VPU_PACKED(bin_header {
    int32_t  magic;
    uint32_t frequency;
};)

void MyriadExecutor::closeDevices(std::vector<DevicePtr> &devicePool) {
    std::lock_guard<std::mutex> lock(device_mutex);
    for (auto &device : devicePool) {
        if (device->_deviceHandle != nullptr) {
            auto res = ncDeviceClose(&(device->_deviceHandle));
            if (res != NC_OK)
                printf("ncDeviceClose failed (%d)\n", static_cast<int>(res));
            device->_deviceHandle = nullptr;
        }
    }
}

void MyriadExecutor::allocateGraph(DevicePtr &device, GraphDesc &graphDesc,
                                   const std::vector<char> &graphFileContent,
                                   const std::pair<const char*, size_t> &graphHeaderDesc,
                                   size_t numStages, const char* networkName) {
    _numStages = numStages;
    if (device->_deviceHandle == nullptr) {
        THROW_IE_EXCEPTION << "Failed to allocate graph: MYRIAD device is not opened.";
    }

    ncStatus_t status;

    status = ncGraphCreate(networkName, &graphDesc._graphHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init graph: " << ncStatusToStr(nullptr, status);
    }

    int executors = device->_platform == MYRIAD_X ? 2 : 1;
    status = ncGraphSetOption(graphDesc._graphHandle, NC_RW_GRAPH_EXECUTORS_NUM, &executors, sizeof(executors));
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to set graph executors: " << ncStatusToStr(nullptr, status);
    }

    status = ncGraphAllocate(device->_deviceHandle,
                             graphDesc._graphHandle,
                             graphFileContent.data(),
                             static_cast<unsigned int>(graphFileContent.size()),
                             graphHeaderDesc.first,
                             graphHeaderDesc.second);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to allocate graph: " << ncStatusToStr(nullptr, status);
    }

    unsigned int dataLength = sizeof(int);

    int numInputs = 0;
    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_INPUT_COUNT, &numInputs, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get number of inputs: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
    if (numInputs != 1) {
        THROW_IE_EXCEPTION << "Unsupported number of inputs: " << numInputs;
    }

    int numOutputs = 0;
    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get number of outputs: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
    if (numOutputs != 1) {
        THROW_IE_EXCEPTION << "Unsupported number of outputs: " << numOutputs;
    }

    dataLength = sizeof(ncTensorDescriptor_t);
    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &graphDesc._inputDesc,
                              &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get input description: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &graphDesc._outputDesc,
                              &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get output description: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    unsigned int fifo_elements = 4;

    status = ncFifoCreate("input", NC_FIFO_HOST_WO, &graphDesc._inputFifoHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init input FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoAllocate(graphDesc._inputFifoHandle, device->_deviceHandle, &graphDesc._inputDesc, fifo_elements);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to create input FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoCreate("output", NC_FIFO_HOST_RO, &graphDesc._outputFifoHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init output FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoAllocate(graphDesc._outputFifoHandle, device->_deviceHandle, &graphDesc._outputDesc, fifo_elements);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to create output FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
}

void MyriadExecutor::queueInference(GraphDesc &graphDesc, void *input_data, size_t input_bytes,
                    void *result_data, size_t result_bytes) {
#ifndef NDEBUG
    if (auto dumpFileName = std::getenv("IE_VPU_DUMP_INPUT_FILE_NAME")) {
        std::ofstream file(dumpFileName, std::ios_base::binary | std::ios_base::out);
        if (!file.is_open()) {
            THROW_IE_EXCEPTION << "[VPU] Cannot open file " << dumpFileName << " for writing";
        }
        file.write(static_cast<const char*>(input_data), input_bytes);
    }
#endif

    if (graphDesc._inputDesc.totalSize != input_bytes) {
        THROW_IE_EXCEPTION << "Input has unexpected size " << input_bytes << ", expected "
                           << graphDesc._inputDesc.totalSize;
    }

    ncStatus_t status = ncGraphQueueInferenceWithFifoElem(graphDesc._graphHandle,
                                graphDesc._inputFifoHandle, graphDesc._outputFifoHandle,
                                input_data, &graphDesc._inputDesc.totalSize, nullptr);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to queue inference: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    if (result_data != nullptr && result_bytes != 0) {
        getResult(graphDesc, result_data, result_bytes);
    }
}

void MyriadExecutor::getResult(GraphDesc &graphDesc, void *result_data, unsigned int result_bytes) {
    ncStatus_t status;
    void *userParam = nullptr;
    status = ncFifoReadElem(graphDesc._outputFifoHandle, result_data, &result_bytes, &userParam);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to read output from FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
}

void MyriadExecutor::deallocateGraph(DevicePtr &device, GraphDesc &graphDesc) {
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
        device->_executors -= 1;
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

std::vector<float> MyriadExecutor::getPerfTimeInfo(ncGraphHandle_t *graphHandle) {
    return getGraphInfo<float>(graphHandle, NC_RO_GRAPH_TIME_TAKEN, _numStages + 2);
}
