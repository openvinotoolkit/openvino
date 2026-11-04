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

// A CUmodule loaded from PTX, unloaded when the last reference goes.
struct cuda_module {
    void* cu_module{nullptr};  // CUmodule (typed as void* to avoid cuda.h in header)

    ~cuda_module();  // defined in sycl_ptx_kernel.cpp to unload the CUmodule
};

// A kernel function looked up in a module.  It holds the module, so the module stays loaded for as
// long as a pending launch holds the function.
struct cuda_function {
    std::shared_ptr<cuda_module> module;
    void* cu_function{nullptr};    // CUfunction
    int max_threads_per_block{0};  // CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK of cu_function
};

// The PTX of one compiled batch and the CUDA modules loaded from it, shared by every entry point of
// the batch and by their clones.  Loading a module JIT-compiles the whole PTX, so each module is
// loaded once for the batch rather than once for every kernel in it.
class ptx_program {
public:
    // One module per dimension mapping: the mapping is baked into the PTX special registers,
    // so a kernel launched through a non-identity mapping needs its own rewritten module.
    static constexpr size_t mapping_count = 6;

    explicit ptx_program(std::string ptx) : _ptx(std::move(ptx)) {}

    const std::string& source() const { return _ptx; }

    // Module for the given dimension mapping, loaded on first use (inside host_task).  requested_by
    // names the kernel that asked for it in error messages.
    std::shared_ptr<cuda_module> module(size_t mapping_idx, const std::string& requested_by);

    // The same module compiled with CU_JIT_THREADS_PER_BLOCK = block_threads (inside host_task).
    std::shared_ptr<cuda_module> capped_module(size_t mapping_idx,
                                               unsigned int block_threads,
                                               const std::string& requested_by);

private:
    const std::string _ptx;

    std::array<std::once_flag, mapping_count> _module_once;
    std::array<std::shared_ptr<cuda_module>, mapping_count> _modules;

    // Register-capped modules, keyed by dimension mapping and block size.
    std::mutex _capped_mutex;
    std::map<std::pair<size_t, unsigned int>, std::shared_ptr<cuda_module>> _capped_modules;
};

// One entry point of a ptx_program and the functions looked up for it.  A launch records a host_task
// that runs after enqueue_kernel() has returned, by which time the caller may have released the
// kernel, so the task holds this state by shared_ptr instead of pointing at the kernel.
struct ptx_kernel_state {
    ptx_kernel_state(std::shared_ptr<ptx_program> program, std::string kernel_name);

    // Function to launch a block of `block_threads` work-items through the given mapping (called
    // inside host_task).  The JIT sizes a kernel's registers without knowing the work-group size,
    // so a register-heavy kernel can come out unable to launch with a large block: at 133
    // registers, 512 work-items need 68096 of the 65536 an A100 multiprocessor has.  When the
    // block is larger than the plain module allows, the function is taken from a module compiled
    // with CU_JIT_THREADS_PER_BLOCK, which keeps ptxas within the registers that block size leaves.
    std::shared_ptr<cuda_function> function_for_launch(size_t mapping_idx, unsigned int block_threads);

    const std::shared_ptr<ptx_program> program;
    const std::string kernel_name;

private:
    std::array<std::once_flag, ptx_program::mapping_count> _function_once;
    std::array<std::shared_ptr<cuda_function>, ptx_program::mapping_count> _functions;

    // Functions from register-capped modules, keyed by dimension mapping and block size.
    std::mutex _capped_mutex;
    std::map<std::pair<size_t, unsigned int>, std::shared_ptr<cuda_function>> _capped_functions;
};

// A compiled PTX kernel with stored argument state.
// CUmodule/CUfunction are loaded lazily on first use inside a SYCL host_task.
class sycl_ptx_kernel : public sycl_base_kernel {
public:
    static constexpr size_t dim_mapping_count = ptx_program::mapping_count;

    // A kernel for the entry point `kernel_name` of `program`.
    sycl_ptx_kernel(std::shared_ptr<ptx_program> program, const std::string& kernel_name, const std::string& kernel_id)
        : _state(std::make_shared<ptx_kernel_state>(std::move(program), kernel_name))
        , _kernel_id(kernel_id) {}

    // A kernel sharing an existing entry point and its looked-up functions (a clone that reuses the handle).
    sycl_ptx_kernel(std::shared_ptr<ptx_kernel_state> state, const std::string& kernel_id)
        : _state(std::move(state))
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

    const std::string& get_ptx_source() const { return _state->program->source(); }
    const std::string& get_kernel_name() const { return _state->kernel_name; }

private:
    std::shared_ptr<ptx_kernel_state> _state;
    std::string _kernel_id;
};

}  // namespace sycl
}  // namespace cldnn
