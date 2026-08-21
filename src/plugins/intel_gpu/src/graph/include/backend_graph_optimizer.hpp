// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine_configuration.hpp"

namespace cldnn {

struct program;

/// Optional graph transformations owned by a runtime backend.
class backend_graph_optimizer {
public:
    virtual ~backend_graph_optimizer() = default;

    virtual void optimize(program& program) const = 0;
};

void register_backend_graph_optimizer(runtime_types runtime, const backend_graph_optimizer& optimizer);
void run_backend_graph_optimizations(program& program);

class backend_graph_optimizer_registration {
public:
    backend_graph_optimizer_registration(runtime_types runtime, const backend_graph_optimizer& optimizer) {
        register_backend_graph_optimizer(runtime, optimizer);
    }
};

}  // namespace cldnn
