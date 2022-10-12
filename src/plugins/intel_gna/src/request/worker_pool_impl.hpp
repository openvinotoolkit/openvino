// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "worker.hpp"
#include "worker_pool.hpp"

namespace GNAPluginNS {
namespace request {

/**
 * @class Implementation of @see WorkerPool interface.
 */
class WorkerPoolImpl : public WorkerPool {
public:
    /**
     * @brief Destroy {WorkerPoolImpl} object
     */
    ~WorkerPoolImpl() override = default;

    /**
     * @see WorkerPool::addModelWorker()
    */
    void addModelWorker(std::shared_ptr<Worker> worker) override;

    /**
     * @see WorkerPool::size()
     */
    size_t size() const override;

    /**
     * @see WorkerPool::empty()
     */
    size_t empty() const override;

    /**
     * @see WorkerPool::worker()
     */
    Worker& worker(uint32_t requestIndex) override;

    /**
     * @see WorkerPool::worker()
     */
    const Worker& worker(uint32_t requestIndex) const override;

    /**
     * @see WorkerPool::firstWorker()
     */
    Worker& firstWorker() override;

    /**
     * @see WorkerPool::firstWorker()
     */
    const Worker& firstWorker() const override;

    /**
     * @see WorkerPool::lastWorker()
     */
    Worker& lastWorker() override;

    /**
     * @see WorkerPool::lastWorker()
     */
    const Worker& lastWorker() const override;

    /**
     * @see WorkerPool::findFreeModelWorker()
     */
    std::shared_ptr<Worker> findFreeModelWorker() override;

private:
    void checkWorkerIndexValid(uint32_t requestIndex) const;
    void checkWorkerNotEmpty() const;

    std::vector<std::shared_ptr<Worker>> modelWorkers_;
};

}  // namespace request
}  // namespace GNAPluginNS
