// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_infer_router.h"
#include "myriad_mvnc_wraper.h"

#include <XLink.h>
#include <mvnc.h>
#include <ncPrivateTypes.h>

#include <vpu/utils/error.hpp>

#include <atomic>
#include <cstring>
#include <limits>

using namespace vpu;
using namespace MyriadPlugin;

static std::atomic<uint32_t> g_channelsCounter {0};

//------------------------------------------------------------------------------
// class InferResultProvider
//------------------------------------------------------------------------------

InferResultProvider::InferResultProvider(
    size_t id, const TensorBuffer& outTensorBuffer):
    m_id(id), m_outputTensor(outTensorBuffer) {}

size_t InferResultProvider::id() const {
    return m_id;
}

InferFuture InferResultProvider::getFuture() {
    return m_promise.get_future();
}

InferPromise& InferResultProvider::getPromise() {
    return m_promise;
}

void InferResultProvider::copyAndSetResult(const streamPacketDesc_t& resultPacket) {
    try {
        VPU_THROW_UNLESS(!m_isCompleted, "Inference's result provider has already received the result.");
        VPU_THROW_UNLESS(resultPacket.length == m_outputTensor.m_length,
                         "Unexpected buffer size. Actual size: %zu; Expect size: %zu",
                         resultPacket.length, m_outputTensor.m_length);

        std::memcpy(m_outputTensor.m_data, resultPacket.data, resultPacket.length);
        m_promise.set_value();
    } catch (...) {
        m_promise.set_exception(std::current_exception());
    }
}

//------------------------------------------------------------------------------
// class MyriadInferRouter
//------------------------------------------------------------------------------

MyriadInferRouter::MyriadInferRouter(
    const Logger::Ptr& log,
    const GraphDesc& graphDesc,
    const ncDeviceHandle_t& deviceHandle,
    uint32_t numElements) :
    m_log(log),
    m_graphDesc(graphDesc),
    m_deviceHandle(deviceHandle) {
    const auto openStream = [ ] (TensorStreamDesc& streamDesc, linkId_t id, const std::string& name, int writeSize) {
        g_channelsCounter++;
        uint32_t channelId = g_channelsCounter.load(std::memory_order_relaxed);

        streamDesc.m_name = name + "Id: " + std::to_string(channelId);
        streamDesc.m_channelId = channelId;
        streamDesc.m_id = XLinkOpenStream(id, streamDesc.m_name.c_str(), writeSize);

        VPU_THROW_UNLESS(streamDesc.m_id != INVALID_STREAM_ID,
            "Fail to open stream. Name: %s", streamDesc.m_name);
    };

    openStream(m_inputStreamDesc, m_deviceHandle.private_data->xlink->linkId,
        "Input tensor stream.", m_graphDesc._inputDesc.totalSize * numElements);
    openStream(m_outputStreamDesc, m_deviceHandle.private_data->xlink->linkId,
        "Output tensor stream.", m_writeSizeStub);

    ncStatus_t nc_rc = ncIOBufferAllocate(m_inputStreamDesc.m_channelId, NC_WRITE,
        m_inputStreamDesc.m_name.c_str(), &m_deviceHandle, &m_graphDesc._inputDesc, numElements);
    VPU_THROW_UNLESS(nc_rc == NC_OK,
                     "Fail to allocate buffer. Stream name: %s", m_inputStreamDesc.m_name);

    nc_rc = ncIOBufferAllocate(m_outputStreamDesc.m_channelId, NC_READ,
        m_outputStreamDesc.m_name.c_str(), &m_deviceHandle, &m_graphDesc._outputDesc, numElements);
    VPU_THROW_UNLESS(nc_rc == NC_OK,
                     "Fail to allocate buffer. Stream name: %s", m_inputStreamDesc.m_name);

    m_receiveResultTask = std::async(std::launch::async, [=]() {
        try {
            while (m_isRunning) {
                // XLink has only blocking operations (there are no "tryDoSomething"
                // or "doSomething(timeout)" methods),
                // so we need logic to know if data should arrive
                if (m_numInfersInProcessing.load(std::memory_order_relaxed)) {
                    m_numInfersInProcessing--;
                    receiveInferResult();
                } else {
                    std::this_thread::sleep_for(m_timeToWaitInferRequestNs);
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lockSendMutex(m_sendRequestMutex);
            m_isRunning = false;

            std::lock_guard<std::mutex> lockMapMutex(m_inferResultsMapMutex);
            for (const auto& providersMapIt : m_idToInferResultMap) {
                auto& resultProvider = providersMapIt.second;
                resultProvider->getPromise().set_exception(std::current_exception());
            }
            m_idToInferResultMap.clear();
        }
    });
}

MyriadInferRouter::~MyriadInferRouter() {
    m_isRunning = false;
    m_receiveResultTask.wait();

    XLinkError_t rc = XLinkWriteData(m_inputStreamDesc.m_id,
        reinterpret_cast<const uint8_t*>(&m_stopSignal), sizeof(m_stopSignal));
    if (rc != X_LINK_SUCCESS) {
        m_log->warning("Fail to send stop signal into input \"%s\", rc = %s",
                       m_inputStreamDesc.m_name, XLinkErrorToStr(rc));
    }

    ncStatus_t ncStatus = ncIOBufferDeallocate(m_inputStreamDesc.m_channelId, &m_deviceHandle);
    if (ncStatus != NC_OK) {
        m_log->warning("Fail to deallocate buffer for channel %s, rc = %s",
            m_inputStreamDesc.m_name, Mvnc::ncStatusToStr(nullptr, ncStatus));
    }

    ncStatus = ncIOBufferDeallocate(m_outputStreamDesc.m_channelId, &m_deviceHandle);
    if (ncStatus != NC_OK) {
        m_log->warning("Fail to deallocate buffer for channel %s, rc = %s",
                       m_outputStreamDesc.m_name, Mvnc::ncStatusToStr(nullptr, ncStatus));
    }

    rc = XLinkCloseStream(m_inputStreamDesc.m_id);
    if (rc != X_LINK_SUCCESS) {
        m_log->warning("Fail to close \"%s\", rc = %s",
                       m_inputStreamDesc.m_name, XLinkErrorToStr(rc));
    }

    rc = XLinkCloseStream(m_outputStreamDesc.m_id);
    if (rc != X_LINK_SUCCESS) {
        m_log->warning("Fail to close \"%s\", rc = %s",
                       m_outputStreamDesc.m_name, XLinkErrorToStr(rc));
    }
}

InferFuture MyriadInferRouter::sendInferAsync(
    const std::vector<uint8_t>& inTensor, const TensorBuffer& outTensorBuffer) {
    std::lock_guard<std::mutex> lockSendMutex(m_sendRequestMutex);

    VPU_THROW_UNLESS(m_isRunning == true, "Myriad infer router was stopped");

    uint32_t requestID = getRequestID();
    XLinkError_t rc = XLinkWriteData(m_inputStreamDesc.m_id,
        reinterpret_cast<const uint8_t*>(&requestID), static_cast<int>(sizeof(requestID)));
    VPU_THROW_UNLESS(rc == X_LINK_SUCCESS, "Failed to send infer request ID by stream %s", m_inputStreamDesc.m_name);

    rc = XLinkWriteData(m_inputStreamDesc.m_id, inTensor.data(), static_cast<int>(inTensor.size()));
    VPU_THROW_UNLESS(rc == X_LINK_SUCCESS, "Failed to send tensor by stream %s", m_inputStreamDesc.m_name);

    ncStatus_t ncRc = ncGraphTrigger(m_graphDesc._graphHandle, m_inputStreamDesc.m_channelId, m_outputStreamDesc.m_channelId);
    VPU_THROW_UNLESS(ncRc == NC_OK, "Failed to send graph trigger command. Graph name: %s", m_graphDesc._name);

    std::lock_guard<std::mutex> lockMapMutex(m_inferResultsMapMutex);
    m_idToInferResultMap[requestID] =
        std::unique_ptr<InferResultProvider>(new InferResultProvider(requestID, outTensorBuffer));
    m_numInfersInProcessing++;

    return m_idToInferResultMap[requestID]->getFuture();
}

uint32_t MyriadInferRouter::getRequestID() {
    VPU_THROW_UNLESS(m_idToInferResultMap.size() != std::numeric_limits<uint32_t>::max(),
        "The maximum number of unprocessed requests has been reached.");

    auto it = m_idToInferResultMap.find(++m_inferRequestCounter);
    while (it != m_idToInferResultMap.end() ||
            m_inferRequestCounter == m_stopSignal) {
        it = m_idToInferResultMap.find(++m_inferRequestCounter);
    }

    return m_inferRequestCounter;
}

void MyriadInferRouter::receiveInferResult() {
    streamPacketDesc_t* packetDesc;
    XLinkError_t rc = XLinkReadData(m_outputStreamDesc.m_id, &packetDesc);
    VPU_THROW_UNLESS(rc == X_LINK_SUCCESS, "Failed to read infer request ID from stream %s", m_outputStreamDesc.m_name);
    VPU_THROW_UNLESS(packetDesc->length == sizeof(uint32_t),
                     "Read wrong packet from stream %s. Expect size: %zu, actual size: %zu",
                     m_inputStreamDesc.m_name, sizeof(uint32_t), packetDesc->length);

    uint32_t requestId = *(reinterpret_cast<const uint32_t*>(packetDesc->data));
    rc = XLinkReleaseData(m_outputStreamDesc.m_id);
    VPU_THROW_UNLESS(rc == X_LINK_SUCCESS,
        "Failed to release data from stream %s", m_outputStreamDesc.m_name);

    std::lock_guard<std::mutex> lockMapMutex(m_inferResultsMapMutex);
    const auto requestPosition = m_idToInferResultMap.find(requestId);
    VPU_THROW_UNLESS(requestPosition != m_idToInferResultMap.end(),
        "Can't find infer result provider by id = %zu", requestId);

    rc = XLinkReadData(m_outputStreamDesc.m_id, &packetDesc);
    VPU_THROW_UNLESS(rc == X_LINK_SUCCESS,
        "Failed to read infer request result from stream %s", m_outputStreamDesc.m_name);

    requestPosition->second->copyAndSetResult(*packetDesc);
    m_idToInferResultMap.erase(requestPosition);

    rc = XLinkReleaseData(m_outputStreamDesc.m_id);
    VPU_THROW_UNLESS(rc == X_LINK_SUCCESS,
        "Failed to release data from stream %s", m_outputStreamDesc.m_name);
}
