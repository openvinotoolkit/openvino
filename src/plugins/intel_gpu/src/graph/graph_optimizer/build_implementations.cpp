// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"

#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

using namespace cldnn;

void build_implementations::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::build_implementations");
    if (p.get_config().get_property(ov::intel_gpu::partial_build_program)) {
        return;
    }

    auto& cache = p.get_kernels_cache();
    for (auto& n : p.get_processing_order()) {
        if (auto impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();
            GPU_DEBUG_TRACE << "add_kernels_source: " << params->desc->id << std::endl;
            cache.add_kernels_source(*params, impl->get_kernels_source());
        }
    }
    cache.build_all();
    for (auto& n : p.get_processing_order()) {
        if (auto impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();
            GPU_DEBUG_TRACE << "init_kernels: " << params->desc->id << std::endl;
            impl->init_kernels(cache, *params);
            impl->reset_kernels_source();
        }
    }
    cache.reset();
}
