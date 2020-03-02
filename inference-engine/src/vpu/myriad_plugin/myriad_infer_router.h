// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <XLinkPublicDefines.h>
#include <mvnc.h>
#include <vpu/utils/logger.hpp>

#include <string>
#include <memory>
#include <mutex>
#include <future>
#include <vector>
#include <unordered_map>

namespace vpu {
namespace MyriadPlugin {

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

using InferFuture = std::future<void>;
using InferPromise = std::promise<void>;

struct TensorBuffer {
    void*    m_data;
    size_t   m_length;
};

struct GraphDesc {
    ncGraphHandle_t* _graphHandle = nullptr;
    std::string _name;

    ncTensorDescriptor_t _inputDesc = {};
    ncTensorDescriptor_t _outputDesc = {};
};

struct TensorStreamDesc {
    uint32_t    m_id;
    uint32_t    m_channelId;
    std::string m_name;
};

//------------------------------------------------------------------------------
// class InferResultProvider
//------------------------------------------------------------------------------

class InferResultProvider {
public:
    using Ptr = std::unique_ptr<InferResultProvider>;

    InferResultProvider(size_t id, const TensorBuffer& outTensorBuffer);

    size_t id() const;
    InferFuture getFuture();
    InferPromise& getPromise();

    void copyAndSetResult(const streamPacketDesc_t& resultPacket);

private:
    size_t m_id;
    bool   m_isCompleted = false;

    InferPromise m_promise;
    TensorBuffer m_outputTensor;
};

//------------------------------------------------------------------------------
// class MyriadInferRouter
//------------------------------------------------------------------------------

class MyriadInferRouter {
public:
    using Ptr = std::shared_ptr<MyriadInferRouter>;

    MyriadInferRouter(
        const Logger::Ptr& log,
        const GraphDesc& graphDesc,
        const ncDeviceHandle_t& deviceHandle,
        uint32_t numElements);

    ~MyriadInferRouter();

    InferFuture sendInferAsync(const std::vector<uint8_t>& inTensor,
        const TensorBuffer& outTensorBuffer);

private:
    uint32_t getRequestID();
    void receiveInferResult();

private:
    Logger::Ptr      m_log;
    GraphDesc        m_graphDesc;
    ncDeviceHandle_t m_deviceHandle;

    bool              m_isRunning = true;
    uint32_t          m_inferRequestCounter {0};
    std::atomic_int   m_numInfersInProcessing {0};
    std::future<void> m_receiveResultTask;

    TensorStreamDesc m_inputStreamDesc;
    TensorStreamDesc m_outputStreamDesc;

    std::mutex m_sendRequestMutex;
    std::mutex m_inferResultsMapMutex;

    std::unordered_map<size_t, InferResultProvider::Ptr> m_idToInferResultMap;

// Constants
    const uint32_t m_stopSignal {0xdead};
    const std::chrono::nanoseconds m_timeToWaitInferRequestNs {10};

    // First, we open streams from the host side,
    // but we can't open a "read-only" stream
    // if the "write" stream did not open from device-side
    const int m_writeSizeStub {8};
};

}  // namespace MyriadPlugin
}  // namespace vpu