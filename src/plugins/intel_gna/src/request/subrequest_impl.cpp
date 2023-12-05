// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subrequest_impl.hpp"

#include <gna2-inference-api.h>

#include "log/debug.hpp"
#include "log/log.hpp"

namespace ov {
namespace intel_gna {
namespace request {

SubrequestImpl::SubrequestImpl(EnqueueHandler enqueueHandler, WaitHandler waitHandler)
    : enqueueHandler_(std::move(enqueueHandler)),
      waitHandler_(std::move(waitHandler)) {
    if (!enqueueHandler_ || !waitHandler_) {
        THROW_GNA_EXCEPTION << "handlers cannot be nullptr";
    }
}

RequestStatus SubrequestImpl::wait(int64_t timeoutMilliseconds) {
    if (!isPending()) {
        return status_;
    }

    try {
        status_ = waitHandler_(requestID_, timeoutMilliseconds);
    } catch (const std::exception& e) {
        ov::intel_gna::log::error() << "Exception when execution wait: " << e.what() << std::endl;
        status_ = RequestStatus::kCompletedWithError;
    }

    return status_;
}

bool SubrequestImpl::enqueue() {
    try {
        requestID_ = enqueueHandler_();
        status_ = RequestStatus::kPending;
    } catch (const std::exception& e) {
        ov::intel_gna::log::error() << "Exception when executiong enqueue: " << e.what() << std::endl;
        status_ = RequestStatus::kCompletedWithError;
    }
    return status_ != RequestStatus::kCompletedWithError;
}

void SubrequestImpl::cleanup() {
    static_cast<void>(wait(0));
    status_ = RequestStatus::kNone;
}

bool SubrequestImpl::isPending() const {
    return status_ == RequestStatus::kPending;
}

bool SubrequestImpl::isAborted() const {
    return status_ == RequestStatus::kAborted;
}

bool SubrequestImpl::isCompleted() const {
    return status_ == RequestStatus::kCompleted;
}

}  // namespace request
}  // namespace intel_gna
}  // namespace ov
