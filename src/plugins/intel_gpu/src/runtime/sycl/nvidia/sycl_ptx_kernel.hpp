// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../sycl_base_kernel.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cldnn {
namespace sycl {

// Assignment of the three NDRange dimensions to the three CUDA axes:
// dim_mapping[d] is the axis (0 = x, 1 = y, 2 = z) that carries NDRange dimension d.
using dim_mapping = std::array<uint8_t, 3>;

// Every permutation of the three axes, identity first.  A kernel is launched through the
// first entry whose block and grid extents satisfy the CUDA limits, so anything that fits
// the natural mapping keeps using it.
const std::array<dim_mapping, 6>& dim_mappings();

// Shared CUDA module/function state — kept alive by shared_ptr for safe capture in host_tasks.
struct cuda_module_state {
    void* cu_module{nullptr};   // CUmodule (typed as void* to avoid cuda.h in header)
    void* cu_function{nullptr}; // CUfunction
    int max_threads_per_block{0};  // CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK of cu_function

    ~cuda_module_state();  // defined in sycl_ptx_kernel.cpp to unload CUmodule
};

// The PTX of a kernel and the CUDA modules lazily loaded from it.  A launch records a host_task
// that runs after enqueue_kernel() has returned, by which time the caller may have released the
// kernel, so the task holds this state by shared_ptr instead of pointing at the kernel.
struct ptx_program_state {
    // One module per dimension mapping: the mapping is baked into the PTX special registers,
    // so a kernel launched through a non-identity mapping needs its own rewritten module.
    static constexpr size_t mapping_count = 6;

    ptx_program_state(std::string ptx, std::string name);

    // Ensure the module for the given dimension mapping is loaded (called inside host_task).
    void ensure_loaded(size_t mapping_idx);

    // Module to launch a block of `block_threads` work-items through the given mapping (called
    // inside host_task).  The JIT sizes a kernel's registers without knowing the work-group size,
    // so a register-heavy kernel can come out unable to launch with a large block: at 133
    // registers, 512 work-items need 68096 of the 65536 an A100 multiprocessor has.  When the
    // block is larger than the plain module allows, the PTX is loaded again with
    // CU_JIT_THREADS_PER_BLOCK, which keeps ptxas within the registers that block size leaves.
    std::shared_ptr<cuda_module_state> module_for_launch(size_t mapping_idx, unsigned int block_threads);

    const std::string ptx_source;
    const std::string kernel_name;

    std::array<std::once_flag, mapping_count> module_once;
    std::array<std::shared_ptr<cuda_module_state>, mapping_count> module_state;

    // Register-capped modules, keyed by dimension mapping and block size.
    std::mutex capped_mutex;
    std::map<std::pair<size_t, unsigned int>, std::shared_ptr<cuda_module_state>> capped_modules;
};

// A compiled PTX kernel with stored argument state.
// CUmodule/CUfunction are loaded lazily on first use inside a SYCL host_task.
class sycl_ptx_kernel : public sycl_base_kernel {
public:
    static constexpr size_t dim_mapping_count = ptx_program_state::mapping_count;

    sycl_ptx_kernel(const std::string& ptx_source, const std::string& kernel_name, const std::string& kernel_id)
        : _program(std::make_shared<ptx_program_state>(ptx_source, kernel_name))
        , _kernel_id(kernel_id) {}

    // --- kernel interface ---
    std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const override;
    bool is_same(const kernel& other) const override;
    std::string get_id() const override { return _kernel_id; }
    std::vector<uint8_t> get_binary() const override;
    std::string get_build_log() const override { return {}; }

    // --- sycl_base_kernel interface ---
    void launch(::sycl::handler& cgh,
                const kernel_arguments_desc& args_desc) override;

    const std::string& get_ptx_source() const { return _program->ptx_source; }
    const std::string& get_kernel_name() const { return _program->kernel_name; }

    // Returns a shared_ptr to module state for safe capture in host_task.
    // Lazily loads CUmodule/CUfunction on first call (must run inside host_task for correct context).
    std::shared_ptr<cuda_module_state> get_module_state(size_t mapping_idx = 0) const {
        return _program->module_state[mapping_idx];
    }

    // See ptx_program_state::ensure_loaded.
    void ensure_loaded(size_t mapping_idx = 0) const { _program->ensure_loaded(mapping_idx); }

    // See ptx_program_state::module_for_launch.
    std::shared_ptr<cuda_module_state> module_for_launch(size_t mapping_idx, unsigned int block_threads) const {
        return _program->module_for_launch(mapping_idx, block_threads);
    }

private:
    std::shared_ptr<ptx_program_state> _program;
    std::string _kernel_id;
};

}  // namespace sycl
}  // namespace cldnn
