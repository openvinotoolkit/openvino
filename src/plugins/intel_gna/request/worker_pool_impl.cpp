// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "worker_pool_impl.hpp"

#include "gna_plugin_log.hpp"
#include "worker.hpp"

namespace GNAPluginNS {
namespace request {

void WorkerPoolImpl::add_model_worker(std::shared_ptr<Worker> worker) {
    worker->set_representing_index(model_workers_.size());
    model_workers_.push_back(std::move(worker));
}

size_t WorkerPoolImpl::size() const {
    return model_workers_.size();
}

size_t WorkerPoolImpl::empty() const {
    return model_workers_.empty();
}

Worker& WorkerPoolImpl::model_worker(uint32_t request_index) {
    return *model_workers_.at(request_index);
}

const Worker& WorkerPoolImpl::model_worker(uint32_t request_index) const {
    check_worker_index_valid(request_index);
    return *model_workers_.at(request_index);
}

Worker& WorkerPoolImpl::first_worker() {
    check_worker_not_empty();
    return *model_workers_.front();
}

const Worker& WorkerPoolImpl::first_worker() const {
    check_worker_not_empty();
    return *model_workers_.front();
}

Worker& WorkerPoolImpl::last_worker() {
    check_worker_not_empty();
    return *model_workers_.back();
}

const Worker& WorkerPoolImpl::last_worker() const {
    check_worker_not_empty();
    return *model_workers_.back();
}

std::shared_ptr<Worker> WorkerPoolImpl::find_free_model_worker() {
    auto free_worker =
        std::find_if(model_workers_.begin(), model_workers_.end(), [](const std::shared_ptr<Worker>& item) {
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

void WorkerPoolImpl::check_worker_index_valid(uint32_t request_index) const {
    if (request_index >= model_workers_.size()) {
        THROW_GNA_EXCEPTION << " no request worker with index: " << request_index;
    }
}

void WorkerPoolImpl::check_worker_not_empty() const {
    if (model_workers_.empty()) {
        THROW_GNA_EXCEPTION << " no request worker created.";
    }
}

}  // namespace request
}  // namespace GNAPluginNS
