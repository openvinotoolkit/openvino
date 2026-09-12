// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend_graph_optimizer.hpp"

#include <map>

#include "intel_gpu/graph/program.hpp"
#include "openvino/core/except.hpp"

namespace cldnn {
namespace {

auto& registered_optimizers() {
    static std::map<runtime_types, const backend_graph_optimizer*> optimizers;
    return optimizers;
}

}  // namespace

void register_backend_graph_optimizer(runtime_types runtime, const backend_graph_optimizer& optimizer) {
    auto& registered = registered_optimizers()[runtime];
    OPENVINO_ASSERT(registered == nullptr || registered == &optimizer, "[GPU] A graph optimizer is already registered for runtime ", runtime);
    registered = &optimizer;
}

bool run_backend_fusion_optimizations(program& program) {
    const auto& optimizers = registered_optimizers();
    const auto optimizer = optimizers.find(program.get_engine().runtime_type());
    if (optimizer != optimizers.end()) {
        return optimizer->second->optimize_fusions(program);
    }
    return false;
}

}  // namespace cldnn
