// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
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

#include "myriad_executor.h"
#include "myriad_config.h"

using namespace vpu::MyriadPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::VPUConfigParams;
using namespace std;
using namespace vpu;

MyriadExecutor::MyriadExecutor(
    const Logger::Ptr& log) :
    _log(log) {}

void MyriadExecutor::allocateGraph(DevicePtr &device, GraphDesc &graphDesc,
                                   const std::vector<char> &graphFileContent,
                                   const std::pair<const char*, size_t> &graphHeaderDesc,
                                   size_t numStages, const std::string & networkName, int executors) {
    VPU_PROFILE(allocateGraph);
    _numStages = numStages;
    graphDesc._name = networkName;
    if (device->_deviceHandle == nullptr) {
        THROW_IE_EXCEPTION << "Failed to allocate graph: MYRIAD device is not opened.";
    }

    ncStatus_t status;

    status = ncGraphCreate(networkName.c_str(), &graphDesc._graphHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init graph: " << Mvnc::ncStatusToStr(nullptr, status);
    }

    status = ncGraphSetOption(graphDesc._graphHandle, NC_RW_GRAPH_EXECUTORS_NUM, &executors, sizeof(executors));
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to set graph executors: " << Mvnc::ncStatusToStr(nullptr, status);
    }

    status = ncGraphAllocate(device->_deviceHandle,
                             graphDesc._graphHandle,
                             graphFileContent.data(),
                             static_cast<unsigned int>(graphFileContent.size()),
                             graphHeaderDesc.first,
                             graphHeaderDesc.second);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to allocate graph: " << Mvnc::ncStatusToStr(nullptr, status);
    }

    unsigned int dataLength = sizeof(int);

    int numInputs = 0;
    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_INPUT_COUNT, &numInputs, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get number of inputs: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
    }
    if (numInputs != 1) {
        THROW_IE_EXCEPTION << "Unsupported number of inputs: " << numInputs;
    }

    int numOutputs = 0;
    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get number of outputs: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
    }
    if (numOutputs != 1) {
        THROW_IE_EXCEPTION << "Unsupported number of outputs: " << numOutputs;
    }

    dataLength = sizeof(ncTensorDescriptor_t);
    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &graphDesc._inputDesc,
                              &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get input description: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncGraphGetOption(graphDesc._graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &graphDesc._outputDesc,
                              &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get output description: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
    }

    unsigned int fifo_elements = (device->_platform == NC_MYRIAD_2 && executors == 1) ? 4 : 2 * executors;

    status = ncFifoCreate("input", NC_FIFO_HOST_WO, &graphDesc._inputFifoHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init input FIFO: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoAllocate(graphDesc._inputFifoHandle, device->_deviceHandle, &graphDesc._inputDesc, fifo_elements);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to create input FIFO: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoCreate("output", NC_FIFO_HOST_RO, &graphDesc._outputFifoHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init output FIFO: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoAllocate(graphDesc._outputFifoHandle, device->_deviceHandle, &graphDesc._outputDesc, fifo_elements);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to create output FIFO: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
    }
}

void MyriadExecutor::queueInference(GraphDesc &graphDesc, void *input_data, size_t input_bytes,
                    void *result_data, size_t result_bytes) {
    VPU_PROFILE(queueInference);
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
        THROW_IE_EXCEPTION << "Failed to queue inference: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
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
        THROW_IE_EXCEPTION << "Failed to read output from FIFO: " << Mvnc::ncStatusToStr(graphDesc._graphHandle, status);
    }
}

void MyriadExecutor::deallocateGraph(DevicePtr &device, GraphDesc &graphDesc) {
    VPU_PROFILE(deallocateGraph);
    if (graphDesc._inputFifoHandle != nullptr) {
        auto res = ncFifoDestroy(&graphDesc._inputFifoHandle);
        if (res != NC_OK)
            _log->warning("ncFifoDelete result %s", Mvnc::ncStatusToStr(nullptr, res));
        graphDesc._inputFifoHandle = nullptr;
    }
    if (graphDesc._outputFifoHandle != nullptr) {
        auto res = ncFifoDestroy(&graphDesc._outputFifoHandle);
        if (res != NC_OK)
            _log->warning("ncFifoDelete result %s", Mvnc::ncStatusToStr(nullptr, res));
        graphDesc._outputFifoHandle = nullptr;
    }
    if (graphDesc._graphHandle != nullptr) {
        auto res = ncGraphDestroy(&graphDesc._graphHandle);
        if (res !=NC_OK)
            _log->warning("Deallocate Graph result %s.", Mvnc::ncStatusToStr(nullptr, res));
        graphDesc._graphHandle = nullptr;
    }
    if (device->_deviceHandle != nullptr) {
        device->_graphNum -= 1;
    }
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
        THROW_IE_EXCEPTION << "Failed to get thermal stats: " << Mvnc::ncStatusToStr(nullptr, status);
    } else {
        return thermal_stats[0];
    }
}

std::vector<float> MyriadExecutor::getPerfTimeInfo(ncGraphHandle_t *graphHandle) {
    return Mvnc::getGraphInfo<float>(graphHandle, NC_RO_GRAPH_TIME_TAKEN, _numStages + 2);
}
