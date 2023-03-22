// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"

#include "intel_gpu/runtime/itt.hpp"

using namespace cldnn;

void build_implementations::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::build_implementations");
    if (p.get_config().get_property(ov::intel_gpu::partial_build_program)) {
        return;
    }

    auto& cache = p.get_kernels_cache();
    cache.build_all();
    for (auto& n : p.get_processing_order()) {
        if (n->get_selected_impl()) {
            n->get_selected_impl()->init_kernels(cache);
            n->get_selected_impl()->reset_kernels_source();
        }
    }
    cache.reset();
}
