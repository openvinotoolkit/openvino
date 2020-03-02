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
    const DevicePtr& device,
    const Logger::Ptr& log) :
    m_device(device),
    m_log(log) {}

void MyriadExecutor::allocateGraph(const std::vector<char> &graphFileContent,
                                   const std::pair<const char*, size_t> &graphHeaderDesc,
                                   size_t numStages, const std::string& networkName, int executors) {
    VPU_PROFILE(allocateGraph);
    m_numStages = numStages;
    m_graphDesc._name = networkName;
    if (m_device->_deviceHandle == nullptr) {
        THROW_IE_EXCEPTION << "Failed to allocate graph: MYRIAD device is not opened.";
    }

    ncStatus_t status;

    status = ncGraphCreate(networkName.c_str(), &m_graphDesc._graphHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init graph: " << Mvnc::ncStatusToStr(nullptr, status);
    }

    status = ncGraphSetOption(m_graphDesc._graphHandle, NC_RW_GRAPH_EXECUTORS_NUM, &executors, sizeof(executors));
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to set graph executors: " << Mvnc::ncStatusToStr(nullptr, status);
    }

    status = ncGraphAllocate(m_device->_deviceHandle,
                             m_graphDesc._graphHandle,
                             graphFileContent.data(),
                             static_cast<unsigned int>(graphFileContent.size()),
                             graphHeaderDesc.first,
                             graphHeaderDesc.second);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to allocate graph: " << Mvnc::ncStatusToStr(nullptr, status);
    }

    unsigned int dataLength = sizeof(int);

    int numInputs = 0;
    status = ncGraphGetOption(m_graphDesc._graphHandle, NC_RO_GRAPH_INPUT_COUNT, &numInputs, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get number of inputs: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }
    if (numInputs != 1) {
        THROW_IE_EXCEPTION << "Unsupported number of inputs: " << numInputs;
    }

    int numOutputs = 0;
    status = ncGraphGetOption(m_graphDesc._graphHandle, NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get number of outputs: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }
    if (numOutputs != 1) {
        THROW_IE_EXCEPTION << "Unsupported number of outputs: " << numOutputs;
    }

    dataLength = sizeof(ncTensorDescriptor_t);
    status = ncGraphGetOption(m_graphDesc._graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &m_graphDesc._inputDesc,
                              &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get input description: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }

    status = ncGraphGetOption(m_graphDesc._graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &m_graphDesc._outputDesc,
                              &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get output description: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }

    unsigned int fifo_elements = (m_device->_platform == NC_MYRIAD_2 && executors == 1) ? 4 : 2 * executors;

    status = ncFifoCreate("input", NC_FIFO_HOST_WO, &m_graphDesc._inputFifoHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init input FIFO: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }

    status = ncFifoAllocate(m_graphDesc._inputFifoHandle, m_device->_deviceHandle, &m_graphDesc._inputDesc, fifo_elements);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to create input FIFO: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }

    status = ncFifoCreate("output", NC_FIFO_HOST_RO, &m_graphDesc._outputFifoHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init output FIFO: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }

    status = ncFifoAllocate(m_graphDesc._outputFifoHandle, m_device->_deviceHandle, &m_graphDesc._outputDesc, fifo_elements);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to create output FIFO: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }
}

void MyriadExecutor::deallocateGraph() {
    VPU_PROFILE(deallocateGraph);
    if (m_graphDesc._inputFifoHandle != nullptr) {
        auto res = ncFifoDestroy(&m_graphDesc._inputFifoHandle);
        if (res != NC_OK)
            m_log->warning("ncFifoDelete result %s", Mvnc::ncStatusToStr(nullptr, res));
        m_graphDesc._inputFifoHandle = nullptr;
    }
    if (m_graphDesc._outputFifoHandle != nullptr) {
        auto res = ncFifoDestroy(&m_graphDesc._outputFifoHandle);
        if (res != NC_OK)
            m_log->warning("ncFifoDelete result %s", Mvnc::ncStatusToStr(nullptr, res));
        m_graphDesc._outputFifoHandle = nullptr;
    }
    if (m_graphDesc._graphHandle != nullptr) {
        auto res = ncGraphDestroy(&m_graphDesc._graphHandle);
        if (res !=NC_OK)
            m_log->warning("Deallocate Graph result %s.", Mvnc::ncStatusToStr(nullptr, res));
        m_graphDesc._graphHandle = nullptr;
    }
    if (m_device->_deviceHandle != nullptr) {
        m_device->_graphNum -= 1;
    }
}

void MyriadExecutor::queueInference(void *input_data, size_t input_bytes,
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

    if (m_graphDesc._inputDesc.totalSize != input_bytes) {
        THROW_IE_EXCEPTION << "Input has unexpected size " << input_bytes << ", expected "
                           << m_graphDesc._inputDesc.totalSize;
    }

    ncStatus_t status = ncGraphQueueInferenceWithFifoElem(m_graphDesc._graphHandle,
                                m_graphDesc._inputFifoHandle, m_graphDesc._outputFifoHandle,
                                input_data, &m_graphDesc._inputDesc.totalSize, nullptr);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to queue inference: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }

    if (result_data != nullptr && result_bytes != 0) {
        getResult(result_data, result_bytes);
    }
}

void MyriadExecutor::getResult(void *result_data, unsigned int result_bytes) {
    ncStatus_t status;
    void *userParam = nullptr;
    status = ncFifoReadElem(m_graphDesc._outputFifoHandle, result_data, &result_bytes, &userParam);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to read output from FIFO: " << Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status);
    }
}

std::vector<float> MyriadExecutor::getPerfTimeInfo() {
    return Mvnc::getGraphInfo<float>(m_graphDesc._graphHandle, NC_RO_GRAPH_TIME_TAKEN, m_numStages + 2);
}
