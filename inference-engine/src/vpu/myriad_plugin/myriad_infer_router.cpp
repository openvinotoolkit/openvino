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
// class MyriadInferRouter::InferResultProvider
//------------------------------------------------------------------------------

class MyriadInferRouter::InferResultProvider {
public:
    using Ptr = std::unique_ptr<InferResultProvider>;

    explicit InferResultProvider(
        const TensorBuffer& outTensorBuffer):
        m_outputTensor(outTensorBuffer) {}

    InferFuture getFuture() {
        return m_promise.get_future();
    }

    InferPromise& getPromise() {
        return m_promise;
    }

    void copyAndSetResult(const inferPacketDesc_t& resultPacket) {
        try {
            VPU_THROW_UNLESS(resultPacket.streamPacket.length == m_outputTensor.m_length,
                             "Unexpected buffer size. Actual size: {}; Expect size: {}",
                             resultPacket.streamPacket.length, m_outputTensor.m_length);

            std::memcpy(m_outputTensor.m_data,
                        resultPacket.streamPacket.data, resultPacket.streamPacket.length);
            m_promise.set_value();
        } catch (...) {
            m_promise.set_exception(std::current_exception());
        }
    }

private:
    InferPromise m_promise;
    TensorBuffer m_outputTensor;
};

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
        "Output tensor stream.", m_writeSizeForReadOnlyStream);

    ncStatus_t nc_rc = ncIOBufferAllocate(m_inputStreamDesc.m_channelId, NC_WRITE,
        m_inputStreamDesc.m_name.c_str(), &m_deviceHandle, &m_graphDesc._inputDesc, numElements);
    VPU_THROW_UNLESS(nc_rc == NC_OK,
                     "Fail to allocate buffer. Stream name: %s", m_inputStreamDesc.m_name);

    nc_rc = ncIOBufferAllocate(m_outputStreamDesc.m_channelId, NC_READ,
        m_outputStreamDesc.m_name.c_str(), &m_deviceHandle, &m_graphDesc._outputDesc, numElements);
    VPU_THROW_UNLESS(nc_rc == NC_OK,
                     "Fail to allocate buffer. Stream name: %s", m_inputStreamDesc.m_name);

    m_receiveResultTask = std::async(std::launch::async, [this]() {
        try {
            while (true) {
                // XLink has only blocking operations (there are no "tryDoSomething"
                // or "doSomething(timeout)" methods),
                // so we need logic to know if data should arrive
                std::unique_lock<std::mutex> lockSendMutex(m_receiveResultMutex);
                m_receiveResultCv.wait(lockSendMutex,
                                       [&]{ return m_numInfersInProcessing != 0 || m_stop; });

                if (m_stop) {
                    break;
                }

                m_numInfersInProcessing--;
                receiveInferResult();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lockSendMutex(m_sendRequestMutex);
                m_stop = true;
            }

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
    m_stop = true;
    m_receiveResultCv.notify_one();
    m_receiveResultTask.wait();

    inferPacketDesc_t packet = { m_stopSignal, {nullptr, 0}};
    XLinkError_t rc = XLinkWriteInferPacket(m_inputStreamDesc.m_id, &packet);
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
    uint32_t requestID = 0;

    {
        std::lock_guard<std::mutex> lockSendMutex(m_sendRequestMutex);

        VPU_THROW_UNLESS(!m_stop, "Myriad infer router was stopped");

        requestID = getRequestID();
        inferPacketDesc_t packet = {
            requestID,
            { const_cast<uint8_t *>(inTensor.data()),
              static_cast<uint32_t>(inTensor.size()) }};

        XLinkError_t rc = XLinkWriteInferPacket(m_inputStreamDesc.m_id, &packet);
        VPU_THROW_UNLESS(rc == X_LINK_SUCCESS, "Failed to send tensor by stream %s", m_inputStreamDesc.m_name);

        ncStatus_t ncRc = ncGraphTrigger(m_graphDesc._graphHandle, m_inputStreamDesc.m_channelId, m_outputStreamDesc.m_channelId);
        VPU_THROW_UNLESS(ncRc == NC_OK, "Failed to send graph trigger command. Graph name: %s", m_graphDesc._name);
    }

    std::lock_guard<std::mutex> lockMapMutex(m_inferResultsMapMutex);
    m_idToInferResultMap[requestID] =
        InferResultProvider::Ptr(new InferResultProvider(outTensorBuffer));

    {
        std::unique_lock<std::mutex> lockSendMutex(m_receiveResultMutex);
        m_numInfersInProcessing++;
        m_receiveResultCv.notify_one();
    }


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
    inferPacketDesc_t* inferPacket = nullptr;
    XLinkError_t rc = XLinkReadDataPacket(
        m_outputStreamDesc.m_id, reinterpret_cast<void**>(&inferPacket));
    VPU_THROW_UNLESS(rc == X_LINK_SUCCESS, "Failed to read data from stream %s", m_outputStreamDesc.m_name);
    uint32_t requestId = inferPacket->id;

    std::lock_guard<std::mutex> lockMapMutex(m_inferResultsMapMutex);
    const auto requestPosition = m_idToInferResultMap.find(requestId);
    VPU_THROW_UNLESS(requestPosition != m_idToInferResultMap.end(),
        "Can't find infer result provider by id = %zu", requestId);

    requestPosition->second->copyAndSetResult(*inferPacket);
    m_idToInferResultMap.erase(requestPosition);

    rc = XLinkReleaseData(m_outputStreamDesc.m_id);
    VPU_THROW_UNLESS(rc == X_LINK_SUCCESS,
        "Failed to release data from stream %s", m_outputStreamDesc.m_name);
}
