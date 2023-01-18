// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "worker_impl.hpp"

#include <gna2-inference-api.h>

#include "log/debug.hpp"
#include "log/log.hpp"
#include "model_wrapper.hpp"
#include "subrequest.hpp"

namespace ov {
namespace intel_gna {
namespace request {

WorkerImpl::WorkerImpl(std::shared_ptr<ModelWrapper> model, std::vector<std::shared_ptr<Subrequest>> modelSubrequests)
    : fullModel_(std::move(model)),
      modelSubrequests_(std::move(modelSubrequests)) {
    if (!fullModel_) {
        THROW_GNA_EXCEPTION << "cannot created request worker for nullptr model";
    }

    if (modelSubrequests_.empty()) {
        THROW_GNA_EXCEPTION << "cannot created request worker for empty subrequest list";
    }

    for (const auto& sunrequest : modelSubrequests_) {
        if (!sunrequest) {
            THROW_GNA_EXCEPTION << "subrequsts cannot be nullptr";
        }
    }
}

const Gna2Model* WorkerImpl::model() const {
    return &fullModel_->object();
}

Gna2Model* WorkerImpl::model() {
    return &fullModel_->object();
}

bool WorkerImpl::enqueueRequest() {
    if (!isFree()) {
        ov::intel_gna::log::warning() << "Trying to propagate on busy request with id: " << representingIndex_;
        return false;
    }

    for (auto& subrequest : modelSubrequests_) {
        if (!subrequest->enqueue()) {
            cleanup_subrequests();
            return false;
        }
    }
    return true;
}

RequestStatus WorkerImpl::wait(int64_t timeoutMilliseconds) {
    bool pending = false;

    // iterate over all configurations for requst
    for (auto& subrequest : modelSubrequests_) {
        if (!subrequest->isPending()) {
            continue;
        }

        auto result = subrequest->wait(timeoutMilliseconds);

        if (result == RequestStatus::kPending) {
            pending = true;
        } else if (result == RequestStatus::kCompletedWithError) {
            cleanup_subrequests();
            return result;
        }
    }

    // return kPending if at least one subrequest is pending
    if (pending) {
        return RequestStatus::kPending;
    }

    // return kAborted if at least one subrequest was aborter
    for (const auto& subrequest : modelSubrequests_) {
        if (subrequest->isAborted()) {
            return RequestStatus::kAborted;
        }
    }

    // return kCompleted if all subrequsts are finish and none of them was aborted
    return RequestStatus::kCompleted;
}

bool WorkerImpl::isFree() const {
    for (const auto& subrequest : modelSubrequests_) {
        if (subrequest->isPending()) {
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

void WorkerImpl::cleanup_subrequests() {
    for (auto& subrequest : modelSubrequests_) {
        if (subrequest->isPending()) {
            subrequest->cleanup();
        }
    }
}

}  // namespace request
}  // namespace intel_gna
}  // namespace ov
