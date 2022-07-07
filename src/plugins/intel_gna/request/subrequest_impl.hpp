// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subrequest.hpp"

namespace GNAPluginNS {
namespace request {

/**
 * @class Implementation of interface @see Subrequest.
 */
class SubrequestImpl : public Subrequest {
public:
    /**
     * @brief Construct {Subrequest}
     * @param enqueueHandler callback to be invoked on enqueue
     * @param enqueueHandler callback to be invoked on wait
     */
    SubrequestImpl(EnqueueHandler enqueueHandler, WaitHandler waitHandler);

    SubrequestImpl(const SubrequestImpl&) = delete;
    SubrequestImpl(SubrequestImpl&&) = delete;
    SubrequestImpl& operator=(const SubrequestImpl&) = delete;
    SubrequestImpl& operator=(SubrequestImpl&&) = delete;

    /**
     * @brief Destroy {SubrequestImpl} object
     */
    ~SubrequestImpl() override = default;

    /**
     * @brief Wait until subrequest will be finished for given timeout.
     * @param timeoutMilliseconds timeout in milliseconds
     * @return status of execution of subrequest @see GNAPluginNS::RequestStatus
     */
    RequestStatus wait(int64_t timeoutMilliseconds) override;

    /**
     * @brief Add subrequest to execution queue.
     */
    void enqueue() override;

    /**
     * @brief Return true if subrequest is pending, otherwise return false
     */
    bool isPending() const override;

    /**
     * @brief Return true if subrequest is aborted, otherwise return false
     */
    bool isAborted() const override;

    /**
     * @brief Return true if subrequest is completed, otherwise return false
     */
    bool isCompleted() const override;

private:
    RequestStatus status_{RequestStatus::kNone};
    uint32_t requestID_{0};
    EnqueueHandler enqueueHandler_;
    WaitHandler waitHandler_;
};

}  // namespace request
}  // namespace GNAPluginNS
