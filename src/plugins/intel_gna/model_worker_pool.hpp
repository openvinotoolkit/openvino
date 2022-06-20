// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "model_worker.hpp"

namespace GNAPluginNS {
class ModelWorkerPool {
public:
    virtual ~ModelWorkerPool() = default;

    virtual void add_model_worker(std::shared_ptr<ModelWorker> model) = 0;

    virtual size_t size() const = 0;
    virtual size_t empty() const = 0;

    virtual ModelWorker& model_worker(uint32_t request_index) = 0;
    virtual const ModelWorker& model_worker(uint32_t request_index) const = 0;
    virtual ModelWorker& first_worker() = 0;
    virtual const ModelWorker& first_worker() const = 0;
    virtual ModelWorker& last_worker() = 0;
    virtual const ModelWorker& last_worker() const = 0;
    virtual std::shared_ptr<ModelWorker> find_free_model_worker() = 0;
};

}  // namespace GNAPluginNS
