// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subrequest.hpp"

#include <gna2-inference-api.h>

#include "gna_plugin_log.hpp"

namespace GNAPluginNS {
namespace request {

Subrequest::Subrequest(EnqueueHandler enqueueHandler, WaitHandler waitHandler)
    : enqueueHandler_(std::move(enqueueHandler)),
      waitHandler_(std::move(waitHandler)) {
    if (!enqueueHandler_ || !waitHandler_) {
        THROW_GNA_EXCEPTION << "handlers cannot be nullptr";
    }
}

RequestStatus Subrequest::wait(int64_t timeoutMilliseconds) {
    if (!isPending()) {
        return status_;
    }

    status_ = waitHandler_(requestID_, timeoutMilliseconds);

    return status_;
}

void Subrequest::enqueue() {
    requestID_ = enqueueHandler_();
    status_ = RequestStatus::kPending;
}

bool Subrequest::isPending() const {
    return status_ == RequestStatus::kPending;
}

bool Subrequest::isAborted() const {
    return status_ == RequestStatus::kAborted;
}

bool Subrequest::isCompleted() const {
    return status_ == RequestStatus::kCompleted;
}

}  // namespace request
}  // namespace GNAPluginNS
