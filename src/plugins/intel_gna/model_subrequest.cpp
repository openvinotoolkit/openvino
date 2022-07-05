// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_subrequest.hpp"

#include <gna2-inference-api.h>

#include "gna_plugin_log.hpp"

namespace GNAPluginNS {

ModelSubrequest::ModelSubrequest(EnqueueHandler enqueue_handler, WaitHandler wait_handler)
    : enqueue_handler_(std::move(enqueue_handler)),
      wait_handler_(std::move(wait_handler)) {
    if (!enqueue_handler_ || !wait_handler_) {
        THROW_GNA_EXCEPTION << "handlers cannot be nullptr";
    }
}

RequestStatus ModelSubrequest::wait(int64_t timeout_miliseconds) {
    if (!is_pending()) {
        return status_;
    }

    status_ = wait_handler_(request_id_, timeout_miliseconds);

    return status_;
}

void ModelSubrequest::enqueue() {
    request_id_ = enqueue_handler_();
    status_ = RequestStatus::kPending;
}

bool ModelSubrequest::is_pending() const {
    return status_ == RequestStatus::kPending;
}

bool ModelSubrequest::is_aborted() const {
    return status_ == RequestStatus::kAborted;
}

bool ModelSubrequest::is_completed() const {
    return status_ == RequestStatus::kCompleted;
}

}  // namespace GNAPluginNS
