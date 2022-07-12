// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subrequest_impl.hpp"

#include <gna2-inference-api.h>

#include "gna_plugin_log.hpp"

namespace GNAPluginNS {
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

    status_ = waitHandler_(requestID_, timeoutMilliseconds);

    return status_;
}

void SubrequestImpl::enqueue() {
    requestID_ = enqueueHandler_();
    status_ = RequestStatus::kPending;
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
}  // namespace GNAPluginNS
