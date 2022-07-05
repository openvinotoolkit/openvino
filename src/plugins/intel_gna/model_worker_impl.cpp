// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_worker_impl.hpp"

#include <gna2-inference-api.h>

#include "gna2_model_wrapper.hpp"
#include "gna_plugin_log.hpp"
#include "model_subrequest.hpp"

namespace GNAPluginNS {

ModelWorkerImpl::ModelWorkerImpl(std::shared_ptr<Gna2ModelWrapper> model,
                                 std::vector<ModelSubrequest> model_subrequests)
    : full_model_(std::move(model)),
      model_subrequests_(std::move(model_subrequests)) {
    if (!full_model_) {
        THROW_GNA_EXCEPTION << "cannot created request worker for nullptr model";
    }

    if (model_subrequests_.empty()) {
        THROW_GNA_EXCEPTION << "cannot created request worker for empty subrequest list";
    }
}

const Gna2Model* ModelWorkerImpl::model() const {
    return &full_model_->object();
}

Gna2Model* ModelWorkerImpl::model() {
    return &full_model_->object();
}

void ModelWorkerImpl::enqueue_request() {
    check_if_free();

    for (auto& subrequest : model_subrequests_) {
        subrequest.enqueue();
    }
}

RequestStatus ModelWorkerImpl::wait(int64_t timeout_miliseconds) {
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

bool ModelWorkerImpl::is_free() const {
    for (const auto& subrequest : model_subrequests_) {
        if (subrequest.is_pending()) {
            return false;
        }
    }

    return true;
}

uint32_t ModelWorkerImpl::representing_index() const {
    return representing_index_;
}

void ModelWorkerImpl::set_representing_index(uint32_t index) {
    representing_index_ = index;
}

void ModelWorkerImpl::set_result(const InferenceEngine::BlobMap& result) {
    request_result_ = result;
}

void ModelWorkerImpl::set_result(InferenceEngine::BlobMap&& result) {
    request_result_ = std::move(result);
}

InferenceEngine::BlobMap& ModelWorkerImpl::result() {
    return request_result_;
}

void ModelWorkerImpl::check_if_free() {
    if (!is_free()) {
        THROW_GNA_EXCEPTION << "Trying to propagte on busy request with id: " << representing_index_;
    }
}

}  // namespace GNAPluginNS
