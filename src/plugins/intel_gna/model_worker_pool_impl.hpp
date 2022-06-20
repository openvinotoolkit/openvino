// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "model_worker.hpp"
#include "model_worker_pool.hpp"

namespace GNAPluginNS {

class ModelWorkerPoolImpl : public ModelWorkerPool {
public:
    void add_model_worker(std::shared_ptr<ModelWorker> worker) override;

    size_t size() const override;
    size_t empty() const override;

    ModelWorker& model_worker(uint32_t request_index) override;
    const ModelWorker& model_worker(uint32_t request_index) const override;
    ModelWorker& first_worker() override;
    const ModelWorker& first_worker() const override;
    ModelWorker& last_worker() override;
    const ModelWorker& last_worker() const override;

    std::shared_ptr<ModelWorker> find_free_model_worker() override;

private:
    void check_worker_index_valid(uint32_t request_index) const;
    void check_worker_not_empty() const;

    std::vector<std::shared_ptr<ModelWorker>> model_workers_;
};

}  // namespace GNAPluginNS
