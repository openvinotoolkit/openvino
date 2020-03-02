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

    auto checkTensor = [](const ncTensorDescriptor_t& tensorDesc) {
        VPU_THROW_UNLESS(tensorDesc.n * tensorDesc.c * tensorDesc.w * tensorDesc.h != 0,
            "Tensor descriptor is invalid. One of the dimensions is zero. n = %zu, c = %zu, w = %zu, h = %zu",
                         tensorDesc.n, tensorDesc.c, tensorDesc.w, tensorDesc.h);

        VPU_THROW_UNLESS(tensorDesc.totalSize != 0,
                         "Tensor descriptor is invalid.  Total size 0");
    };

    dataLength = sizeof(ncTensorDescriptor_t);
    status = ncGraphGetOption(m_graphDesc._graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &m_graphDesc._inputDesc,
                              &dataLength);

    VPU_THROW_UNLESS(status == NC_OK, "Failed to get input description: %s",
        Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status));
    checkTensor(m_graphDesc._inputDesc);

    status = ncGraphGetOption(m_graphDesc._graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &m_graphDesc._outputDesc,
                              &dataLength);

    VPU_THROW_UNLESS(status == NC_OK, "Failed to get output description: %s",
                     Mvnc::ncStatusToStr(m_graphDesc._graphHandle, status));
    checkTensor(m_graphDesc._outputDesc);

    unsigned int numElements = (m_device->_platform == NC_MYRIAD_2 && executors == 1) ? 4 : 2 * executors;

    m_infersRouter = std::make_shared<MyriadInferRouter>(
        m_log, m_graphDesc, *m_device->_deviceHandle, numElements);
}

void MyriadExecutor::deallocateGraph() {
    VPU_PROFILE(deallocateGraph);
    m_infersRouter = nullptr;

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

InferFuture MyriadExecutor::sendInferAsync(
    const std::vector<uint8_t>& inTensor, const TensorBuffer& outTensorBuffer) {
    VPU_PROFILE(sendInferAsync);
#ifndef NDEBUG
    if (auto dumpFileName = std::getenv("IE_VPU_DUMP_INPUT_FILE_NAME")) {
        std::ofstream file(dumpFileName, std::ios_base::binary | std::ios_base::out);
        if (!file.is_open()) {
            THROW_IE_EXCEPTION << "[VPU] Cannot open file " << dumpFileName << " for writing";
        }
        file.write(reinterpret_cast<const char*>(inTensor.data()), inTensor.size());
    }
#endif

    return m_infersRouter->sendInferAsync(inTensor, outTensorBuffer);
}

std::vector<float> MyriadExecutor::getPerfTimeInfo() {
    return Mvnc::getGraphInfo<float>(m_graphDesc._graphHandle, NC_RO_GRAPH_TIME_TAKEN, m_numStages + 2);
}
