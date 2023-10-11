// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "worker_pool_impl.hpp"

#include "log/debug.hpp"
#include "worker.hpp"

namespace ov {
namespace intel_gna {
namespace request {

void WorkerPoolImpl::addModelWorker(std::shared_ptr<Worker> worker) {
    if (!worker) {
        THROW_GNA_EXCEPTION << "cannot not add nullptr request worker to the pool";
    }
    worker->setRepresentingIndex(static_cast<uint32_t>(modelWorkers_.size()));
    modelWorkers_.push_back(std::move(worker));
}

size_t WorkerPoolImpl::size() const {
    return modelWorkers_.size();
}

size_t WorkerPoolImpl::empty() const {
    return modelWorkers_.empty();
}

Worker& WorkerPoolImpl::worker(uint32_t requestIndex) {
    return *modelWorkers_.at(requestIndex);
}

const Worker& WorkerPoolImpl::worker(uint32_t requestIndex) const {
    checkWorkerIndexValid(requestIndex);
    return *modelWorkers_.at(requestIndex);
}

Worker& WorkerPoolImpl::firstWorker() {
    checkWorkerNotEmpty();
    return *modelWorkers_.front();
}

const Worker& WorkerPoolImpl::firstWorker() const {
    checkWorkerNotEmpty();
    return *modelWorkers_.front();
}

Worker& WorkerPoolImpl::lastWorker() {
    checkWorkerNotEmpty();
    return *modelWorkers_.back();
}

const Worker& WorkerPoolImpl::lastWorker() const {
    checkWorkerNotEmpty();
    return *modelWorkers_.back();
}

std::shared_ptr<Worker> WorkerPoolImpl::findFreeModelWorker() {
    auto freeWorker = std::find_if(modelWorkers_.begin(), modelWorkers_.end(), [](const std::shared_ptr<Worker>& item) {
        if (item) {
            return item->isFree();
        }
        return false;
    });
    if (freeWorker != modelWorkers_.end()) {
        return *freeWorker;
    }
    return nullptr;
}

void WorkerPoolImpl::checkWorkerIndexValid(uint32_t requestIndex) const {
    if (requestIndex >= modelWorkers_.size()) {
        THROW_GNA_EXCEPTION << " no request worker with index: " << requestIndex;
    }
}

void WorkerPoolImpl::checkWorkerNotEmpty() const {
    if (modelWorkers_.empty()) {
        THROW_GNA_EXCEPTION << " no request worker created.";
    }
}

}  // namespace request
}  // namespace intel_gna
}  // namespace ov
