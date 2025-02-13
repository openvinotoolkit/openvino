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

    std::cout << __FILE__ << ":" << __LINE__ << " === " << " p.get_kernels_cache()" << std::endl;
    auto& cache = p.get_kernels_cache();
    std::cout << __FILE__ << ":" << __LINE__ << " === " << " add_kernels_source" << std::endl;
    for (auto& n : p.get_processing_order()) {
        if (auto impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();
            cache.add_kernels_source(*params, impl->get_kernels_source());
        }
    }
    std::cout << __FILE__ << ":" << __LINE__ << " === " << " build_all" << std::endl;
    cache.build_all();
    std::cout << __FILE__ << ":" << __LINE__ << " === " << " init_kernels" << std::endl;
    for (auto& n : p.get_processing_order()) {
        if (auto impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();
            impl->init_kernels(cache, *params);
            impl->reset_kernels_source();
        }
    }
    std::cout << __FILE__ << ":" << __LINE__ << " === " << " cache.reset()" << std::endl;
    cache.reset();
    std::cout << __FILE__ << ":" << __LINE__ << " === " << " complete build_implementations::run" << std::endl;
}
