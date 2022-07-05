// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "worker_impl.hpp"

#include <gna2-inference-api.h>

#include "gna_plugin_log.hpp"
#include "model_wrapper.hpp"
#include "subrequest.hpp"

namespace GNAPluginNS {
namespace request {

WorkerImpl::WorkerImpl(std::shared_ptr<ModelWrapper> model, std::vector<Subrequest> model_subrequests)
    : full_model_(std::move(model)),
      model_subrequests_(std::move(model_subrequests)) {
    if (!full_model_) {
        THROW_GNA_EXCEPTION << "cannot created request worker for nullptr model";
    }

    if (model_subrequests_.empty()) {
        THROW_GNA_EXCEPTION << "cannot created request worker for empty subrequest list";
    }
}

const Gna2Model* WorkerImpl::model() const {
    return &full_model_->object();
}

Gna2Model* WorkerImpl::model() {
    return &full_model_->object();
}

void WorkerImpl::enqueue_request() {
    check_if_free();

    for (auto& subrequest : model_subrequests_) {
        subrequest.enqueue();
    }
}

RequestStatus WorkerImpl::wait(int64_t timeout_miliseconds) {
    bool pending = false;

    // iterate over all configurations for requst
    for (auto& subrequest : model_subrequests_) {
        if (!subrequest.is_pending()) {
            continue;
        }

        if (subrequest.wait(timeout_miliseconds) == RequestStatus::kPending) {
            pending = true;
        }
    }

    // return kPending if at least one subrequest is pending
    if (pending) {
        return RequestStatus::kPending;
    }

    // return kAborted if at least one subrequest was aborter
    for (const auto& subrequest : model_subrequests_) {
        if (subrequest.is_aborted()) {
            return RequestStatus::kAborted;
        }
    }

    // return kCompleted if all subrequsts are finish and none of them was aborted
    return RequestStatus::kCompleted;
}

bool WorkerImpl::is_free() const {
    for (const auto& subrequest : model_subrequests_) {
        if (subrequest.is_pending()) {
            return false;
        }
    }

    return true;
}

uint32_t WorkerImpl::representing_index() const {
    return representing_index_;
}

void WorkerImpl::set_representing_index(uint32_t index) {
    representing_index_ = index;
}

void WorkerImpl::set_result(const InferenceEngine::BlobMap& result) {
    request_result_ = result;
}

void WorkerImpl::set_result(InferenceEngine::BlobMap&& result) {
    request_result_ = std::move(result);
}

InferenceEngine::BlobMap& WorkerImpl::result() {
    return request_result_;
}

void WorkerImpl::check_if_free() {
    if (!is_free()) {
        THROW_GNA_EXCEPTION << "Trying to propagte on busy request with id: " << representing_index_;
    }
}

}  // namespace request
}  // namespace GNAPluginNS
