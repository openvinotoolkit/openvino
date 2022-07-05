// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>

#include "request_status.hpp"

namespace GNAPluginNS {
namespace request {

// TODO move/copy or pointer?
class Subrequest {
public:
    /**
     * @brief Callback invoked by enqueue operation.
     * @return request id
     */
    using EnqueueHandler = std::function<uint32_t()>;

    /**
     * @brief Callback invoked by wait operation.
     * @param requestID id of request to be used for wait
     * @param timeoutMilliseconds timeout of wait in milliseconds
     * @return Status of subrequest @see GNAPluginNS::RequestStatus
     */
    using WaitHandler = std::function<RequestStatus(uint32_t requestID, int64_t timeoutMilliseconds)>;

    /**
     * @brief Construct {Subrequest}
     * @param enqueueHandler callback to be invoked on enqueue
     * @param enqueueHandler callback to be invoked on wait
     */
    Subrequest(EnqueueHandler enqueueHandler, WaitHandler waitHandler);

    Subrequest(const Subrequest&) = default;
    Subrequest(Subrequest&&) = default;
    Subrequest& operator=(const Subrequest&) = default;
    Subrequest& operator=(Subrequest&&) = default;

    /**
    * @brief Wait until subrequest will be finished for given timeout.
    * @param timeoutMilliseconds timeout in milliseconds
    * @return status of execution of subrequest @see GNAPluginNS::RequestStatus
    */
    RequestStatus wait(int64_t timeoutMilliseconds);

    /**
    * @brief Add subrequest to execution queue.
    */
    void enqueue();

    /**
    * @brief Return true if subrequest is pending, otherwise return false
    */
    bool isPending() const;

    /**
     * @brief Return true if subrequest is aborted, otherwise return false
     */
    bool isAborted() const;

    /**
     * @brief Return true if subrequest is completed, otherwise return false
     */
    bool isCompleted() const;

private:
    RequestStatus status_{RequestStatus::kNone};
    uint32_t requestID_{0};
    EnqueueHandler enqueueHandler_;
    WaitHandler waitHandler_;
};

}  // namespace request
}  // namespace GNAPluginNS
