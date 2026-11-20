// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_queries.hpp"

#include <map>
#include <memory>
#include <mutex>

#include "impls/ocl/kernels_cache.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"

namespace cldnn {

bool query_register_file_size_option_supported(cldnn::engine& e, const cldnn::ExecutionConfig& config) {
    auto* device = e.get_device().get();
    if (device->get_info().arch < gpu_arch::xe3) {
        return false;
    }

    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    static std::map<cldnn::device*, bool> cache;
    if (cache.find(device) != cache.end()) {
        return cache.at(device);
    }

    auto kernel_string = std::make_shared<cldnn::kernel_string>();
    kernel_string->str = "kernel void reg_file_size_check() {}";
    kernel_string->options = "-ze-exp-register-file-size 128";
    kernel_string->entry_point = "reg_file_size_check";
    kernel_string->batch_compilation = true;

    try {
        cldnn::kernel_impl_params dummy_params;
        auto kernels_cache_device_query = std::make_unique<cldnn::kernels_cache>(e, config, 0);
        kernels_cache_device_query->add_kernels_source(dummy_params, {kernel_string}, false);
        kernels_cache_device_query->build_all();
        cache[device] = true;
    } catch (std::exception&) {
        cache[device] = false;
    }

    return cache.at(device);
}

}  // namespace cldnn
