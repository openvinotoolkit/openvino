// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_subrequest.hpp"

#include "gna_device_interface.hpp"
#include "gna_plugin_log.hpp"
#include <gna2-inference-api.h>

namespace GNAPluginNS {

ModelSubrequest::ModelSubrequest(uint64_t request_config_id, EnqueueHandler enqueue_handler, WaitHandler wait_handler)
    : request_config_id_(request_config_id),
      enqueue_handler_(std::move(enqueue_handler)),
      wait_handler_(std::move(wait_handler)) {
    if (!enqueue_handler_ || !wait_handler_) {
        THROW_GNA_EXCEPTION << "handlers cannot be nullptr";
    }
}

ModelSubrequest::Status ModelSubrequest::wait(int64_t timeout_miliseconds) {
    if (!is_ongoing()) {
        return status_;
    }

    GNARequestWaitStatus wait_status = wait_handler_(request_id(), timeout_miliseconds);

    if (wait_status == GNARequestWaitStatus::kPending) {
        status_ = Status::kOngoing;
    } else if (wait_status == GNARequestWaitStatus::kAborted) {
        status_ = Status::kAborted;
    } else {
        status_ = Status::kCompleted;
    }

    return status_;
}

void ModelSubrequest::enqueue() {
    request_id_ = enqueue_handler_(request_config_id_);
    status_ = Status::kOngoing;
}

uint64_t ModelSubrequest::request_config_id() const {
    return request_config_id_;
}

uint32_t ModelSubrequest::request_id() const {
    return request_id_;
}

bool ModelSubrequest::is_ongoing() const {
    return status_ == Status::kOngoing;
}

bool ModelSubrequest::is_aborted() const {
    return status_ == Status::kAborted;
}

bool ModelSubrequest::is_ready() const {
    return status_ == Status::kReady;
}

bool ModelSubrequest::is_completed() const {
    return status_ == Status::kCompleted;
}

}  // namespace GNAPluginNS
