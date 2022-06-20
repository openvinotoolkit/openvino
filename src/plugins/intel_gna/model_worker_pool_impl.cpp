// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_worker_pool_impl.hpp"

#include "gna_plugin_log.hpp"
#include "model_worker.hpp"

namespace GNAPluginNS {

void ModelWorkerPoolImpl::add_model_worker(std::shared_ptr<ModelWorker> worker) {
    worker->set_representing_index(model_workers_.size());
    model_workers_.push_back(std::move(worker));
}

size_t ModelWorkerPoolImpl::size() const {
    return model_workers_.size();
}

size_t ModelWorkerPoolImpl::empty() const {
    return model_workers_.empty();
}

ModelWorker& ModelWorkerPoolImpl::model_worker(uint32_t request_index) {
    return *model_workers_.at(request_index);
}

const ModelWorker& ModelWorkerPoolImpl::model_worker(uint32_t request_index) const {
    check_worker_index_valid(request_index);
    return *model_workers_.at(request_index);
}

ModelWorker& ModelWorkerPoolImpl::first_worker() {
    check_worker_not_empty();
    return *model_workers_.front();
}

const ModelWorker& ModelWorkerPoolImpl::first_worker() const {
    check_worker_not_empty();
    return *model_workers_.front();
}

ModelWorker& ModelWorkerPoolImpl::last_worker() {
    check_worker_not_empty();
    return *model_workers_.back();
}

const ModelWorker& ModelWorkerPoolImpl::last_worker() const {
    check_worker_not_empty();
    return *model_workers_.back();
}

std::shared_ptr<ModelWorker> ModelWorkerPoolImpl::find_free_model_worker() {
    auto free_worker =
        std::find_if(model_workers_.begin(), model_workers_.end(), [](const std::shared_ptr<ModelWorker>& item) {
            if (item) {
                return item->is_free();
            }
            return false;
        });
    if (free_worker != model_workers_.end()) {
        return *free_worker;
    }
    return nullptr;
}

void ModelWorkerPoolImpl::check_worker_index_valid(uint32_t request_index) const {
    if (request_index >= model_workers_.size()) {
        THROW_GNA_EXCEPTION << " no request worker with index: " << request_index;
    }
}

void ModelWorkerPoolImpl::check_worker_not_empty() const {
    if (model_workers_.empty()) {
        THROW_GNA_EXCEPTION << " no request worker created.";
    }
}

}  // namespace GNAPluginNS
