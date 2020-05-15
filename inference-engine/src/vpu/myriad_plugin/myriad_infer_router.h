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
// class MyriadInferRouter
//------------------------------------------------------------------------------

class MyriadInferRouter {
public:
    using Ptr = std::unique_ptr<MyriadInferRouter>;

    MyriadInferRouter(
        const Logger::Ptr& log,
        const GraphDesc& graphDesc,
        const ncDeviceHandle_t& deviceHandle,
        uint32_t numElements);

    ~MyriadInferRouter();

    InferFuture sendInferAsync(const std::vector<uint8_t>& inTensor,
        const TensorBuffer& outTensorBuffer);

private:
    class InferResultProvider;

private:
    uint32_t getRequestID();
    void receiveInferResult();

private:
    Logger::Ptr      m_log;
    GraphDesc        m_graphDesc;
    ncDeviceHandle_t m_deviceHandle;

    bool     m_stop = false;
    uint32_t m_inferRequestCounter {0};
    int      m_numInfersInProcessing {0};

    std::future<void> m_receiveResultTask;

    TensorStreamDesc m_inputStreamDesc;
    TensorStreamDesc m_outputStreamDesc;

    std::mutex m_sendRequestMutex;
    std::mutex m_inferResultsMapMutex;

    std::mutex m_receiveResultMutex;
    std::condition_variable m_receiveResultCv;

    std::unordered_map<size_t, std::unique_ptr<InferResultProvider>> m_idToInferResultMap;

    // It is impossible to open "read-only" stream from host, so we need to set some value for write size.
    // It cannot be zero as well.
    // Use this constant as write size to make an intention to open "read-only" stream.
    static constexpr int m_writeSizeForReadOnlyStream {8};
    static constexpr uint32_t m_stopSignal {0xdead};
};

}  // namespace MyriadPlugin
}  // namespace vpu
