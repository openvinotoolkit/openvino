// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "worker.hpp"

namespace GNAPluginNS {
namespace request {

class WorkerPool {
public:
    virtual ~WorkerPool() = default;

    virtual void add_model_worker(std::shared_ptr<Worker> model) = 0;

    virtual size_t size() const = 0;
    virtual size_t empty() const = 0;

    virtual Worker& model_worker(uint32_t request_index) = 0;
    virtual const Worker& model_worker(uint32_t request_index) const = 0;
    virtual Worker& first_worker() = 0;
    virtual const Worker& first_worker() const = 0;
    virtual Worker& last_worker() = 0;
    virtual const Worker& last_worker() const = 0;
    virtual std::shared_ptr<Worker> find_free_model_worker() = 0;
};

}  // namespace request
}  // namespace GNAPluginNS
