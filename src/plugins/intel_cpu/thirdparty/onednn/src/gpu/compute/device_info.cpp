/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <mutex>
#include <thread>

#include "gpu/compute/device_info.hpp"

#include "common/verbose.hpp"
#include "gpu/jit/binary_format.hpp"

#ifdef DNNL_WITH_SYCL
#include "sycl/sycl_engine_base.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

uint64_t get_future_extensions(compute::gpu_arch_t gpu_arch) {
    using namespace compute;

    uint64_t extensions = 0;
    switch (gpu_arch) {
        case gpu_arch_t::xe_hp:
        case gpu_arch_t::xe_hpg:
            extensions |= (uint64_t)device_ext_t::intel_global_float_atomics;
            extensions |= (uint64_t)
                    device_ext_t::intel_subgroup_matrix_multiply_accumulate;
            extensions |= (uint64_t)device_ext_t::
                    intel_subgroup_split_matrix_multiply_accumulate;
            extensions
                    |= (uint64_t)device_ext_t::intel_variable_eu_thread_count;
            extensions |= (uint64_t)device_ext_t::future_bf16_cvt;
        case gpu_arch_t::xe_lp:
            extensions |= (uint64_t)device_ext_t::intel_dot_accumulate;
            break;
        default: break;
    }
    return extensions;
}

inline gpu_arch_t str2gpu_arch(const char *str) {
#define CASE(_case) \
    if (!strcmp(STRINGIFY(_case), str)) return gpu_arch_t::_case

    CASE(gen9);
    CASE(xe_lp);
    CASE(xe_hp);
    CASE(xe_hpg);
    return gpu_arch_t::unknown;
#undef CASE
}

bool device_info_t::mayiuse_ngen_kernels(engine_t *engine) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    if (checked_ngen_kernels_) return mayiuse_ngen_kernels_;

    auto status
            = jit::gpu_supports_binary_format(&mayiuse_ngen_kernels_, engine);
    if (status != status::success) mayiuse_ngen_kernels_ = false;

    if (get_verbose())
        printf("dnnl_verbose,info,gpu,binary_kernels:%s\n",
                mayiuse_ngen_kernels_ ? "enabled" : "disabled");

    checked_ngen_kernels_ = true;

    return mayiuse_ngen_kernels_;
}

bool device_info_t::mayiuse_sub_group(int size) const {
    return utils::one_of(size, 8, 16, 32);
}

status_t device_info_t::init_attributes_common(engine_t *engine) {
    // TODO: Fix for discrete GPUs. The code below is written for
    // integrated GPUs assuming that last-level cache for GPU is shared
    // with CPU.
    // Integrated GPUs share LLC with CPU which is L3 cache on CPU.

    // XXX: this is the only place where GPU runtime functionally depends on
    // CPU runtime. The `llc_cache_size_` is used only in one kernel for gen9.
    // The idea is to use approximate cache size.

    // llc_cache_size_ = cpu::platform::get_per_core_cache_size(3)
    //        * cpu::platform::get_num_cores();
    // Assumption is that HT is likely enabled on client systems.
    llc_cache_size_ = std::thread::hardware_concurrency() * (1 << 20);

    // Assume 7 threads by default
    int32_t threads_per_eu[2] = {7, 7};
    switch (gpu_arch_) {
        case gpu::compute::gpu_arch_t::gen9:
        case gpu::compute::gpu_arch_t::xe_lp:
            threads_per_eu[0] = 7;
            threads_per_eu[1] = 7;
            break;
        case gpu::compute::gpu_arch_t::xe_hp:
        case gpu::compute::gpu_arch_t::xe_hpg:
            threads_per_eu[0] = 8; // 128 regs/thread
            threads_per_eu[1] = 4; // 256 regs/thread
            break;
        default: break;
    }

    hw_threads_[0] = eu_count_ * threads_per_eu[0];
    hw_threads_[1] = eu_count_ * threads_per_eu[1];

    max_eus_per_wg_ = 8;
    switch (gpu_arch_) {
        case gpu::compute::gpu_arch_t::gen9: max_eus_per_wg_ = 8; break;
        case gpu::compute::gpu_arch_t::xe_lp:
        case gpu::compute::gpu_arch_t::xe_hp:
        case gpu::compute::gpu_arch_t::xe_hpg: max_eus_per_wg_ = 16; break;
        default: break;
    }

    mayiuse_non_uniform_work_groups_ = true;
#ifdef DNNL_WITH_SYCL
    if (engine->runtime_kind() == runtime_kind::sycl) {
        auto *sycl_engine
                = utils::downcast<const sycl::sycl_engine_base_t *>(engine);
        // Level Zero backend does not support non-uniform work-groups.
        mayiuse_non_uniform_work_groups_
                = (sycl_engine->backend() == sycl::backend_t::opencl);
    }
#endif

    return status::success;
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl
