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

WorkerImpl::WorkerImpl(std::shared_ptr<ModelWrapper> model, std::vector<Subrequest> modelSubrequests)
    : fullModel_(std::move(model)),
      modelSubrequests_(std::move(modelSubrequests)) {
    if (!fullModel_) {
        THROW_GNA_EXCEPTION << "cannot created request worker for nullptr model";
    }

    if (modelSubrequests_.empty()) {
        THROW_GNA_EXCEPTION << "cannot created request worker for empty subrequest list";
    }
}

const Gna2Model* WorkerImpl::model() const {
    return &fullModel_->object();
}

Gna2Model* WorkerImpl::model() {
    return &fullModel_->object();
}

void WorkerImpl::enqueueRequest() {
    check_if_free();

    for (auto& subrequest : modelSubrequests_) {
        subrequest.enqueue();
    }
}

RequestStatus WorkerImpl::wait(int64_t timeoutMilliseconds) {
    bool pending = false;

    // iterate over all configurations for requst
    for (auto& subrequest : modelSubrequests_) {
        if (!subrequest.isPending()) {
            continue;
        }

        if (subrequest.wait(timeoutMilliseconds) == RequestStatus::kPending) {
            pending = true;
        }
    }

    // return kPending if at least one subrequest is pending
    if (pending) {
        return RequestStatus::kPending;
    }

    // return kAborted if at least one subrequest was aborter
    for (const auto& subrequest : modelSubrequests_) {
        if (subrequest.isAborted()) {
            return RequestStatus::kAborted;
        }
    }

    // return kCompleted if all subrequsts are finish and none of them was aborted
    return RequestStatus::kCompleted;
}

bool WorkerImpl::isFree() const {
    for (const auto& subrequest : modelSubrequests_) {
        if (subrequest.isPending()) {
            return false;
        }
    }

    return true;
}

uint32_t WorkerImpl::representingIndex() const {
    return representingIndex_;
}

void WorkerImpl::setRepresentingIndex(uint32_t index) {
    representingIndex_ = index;
}

void WorkerImpl::setResult(const InferenceEngine::BlobMap& result) {
    requestResult_ = result;
}

void WorkerImpl::setResult(InferenceEngine::BlobMap&& result) {
    requestResult_ = std::move(result);
}

InferenceEngine::BlobMap& WorkerImpl::result() {
    return requestResult_;
}

void WorkerImpl::check_if_free() {
    if (!isFree()) {
        THROW_GNA_EXCEPTION << "Trying to propagte on busy request with id: " << representingIndex_;
    }
}

}  // namespace request
}  // namespace GNAPluginNS
