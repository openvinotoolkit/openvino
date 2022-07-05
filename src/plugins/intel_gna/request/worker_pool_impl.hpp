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

class WorkerPoolImpl : public WorkerPool {
public:
    void add_model_worker(std::shared_ptr<Worker> worker) override;

    size_t size() const override;
    size_t empty() const override;

    Worker& model_worker(uint32_t request_index) override;
    const Worker& model_worker(uint32_t request_index) const override;
    Worker& first_worker() override;
    const Worker& first_worker() const override;
    Worker& last_worker() override;
    const Worker& last_worker() const override;

    std::shared_ptr<Worker> find_free_model_worker() override;

private:
    void check_worker_index_valid(uint32_t request_index) const;
    void check_worker_not_empty() const;

    std::vector<std::shared_ptr<Worker>> model_workers_;
};

}  // namespace request
}  // namespace GNAPluginNS
