// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "worker.hpp"

namespace GNAPluginNS {
namespace request {

/**
 * @interface Interface of pool of workers. Interface allows to add and retrieve the workers.
 */
class WorkerPool {
public:
    /**
     * Destruct {WorkerPool} object
     */
    virtual ~WorkerPool() = default;

    /**
     * @brief Add worker to the pool. If more than one worker is added more than request can be queued at the same time.
     * @worker pointer to worker to tbe added to the pool.
     */
    virtual void addModelWorker(std::shared_ptr<Worker> worker) = 0;

    /**
     * @brief Return number of workers in the pool.
     */
    virtual size_t size() const = 0;

    /**
     * @brief Return true if there is no workers in the pool, otherwise return false
     */
    virtual size_t empty() const = 0;

    /**
     * @brief Return worker for given index.
     * @param index of worker to be returned
     * @return reference to the worker for index.
     * @throw exception in case index is invalid
     */
    virtual Worker& worker(uint32_t index) = 0;

    /**
     * @brief Return worker for given index.
     * @param index of worker to be returned
     * @return reference to the worker for index.
     * @throw exception in case index is invalid
     */
    virtual const Worker& worker(uint32_t index) const = 0;

    /**
     * @brief Return worker which was added to the pool as the first one
     * @throw exception in case no worker in pool
     */
    virtual Worker& firstWorker() = 0;

    /**
     * @brief Return worker which was added to the pool as the first one
     * @throw exception in case no worker in pool
     */
    virtual const Worker& firstWorker() const = 0;

    /**
     * @brief Return worker which was added to the pool as the last one
     * @throw exception in case no worker in pool
     */
    virtual Worker& lastWorker() = 0;

    /**
     * @brief Return worker which was added to the pool the last one
     * @throw exception in case no worker in pool
     */
    virtual const Worker& lastWorker() const = 0;

    /**
     * @brief Return worker which is not busy.
     * @return pointer to free worker, nullptr in case no free worker was found.
     */
    virtual std::shared_ptr<Worker> findFreeModelWorker() = 0;
};

}  // namespace request
}  // namespace GNAPluginNS
