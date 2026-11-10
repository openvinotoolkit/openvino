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

// A CUmodule loaded from PTX, unloaded when the last reference goes.  That can be on any thread - an
// application thread releasing the last kernel, with no context or another one current - so the module
// remembers the context it was loaded into and unloads from it.  When that is the device's primary
// context, the module also holds a reference to it, so the context outlives the module.
struct cuda_module {
    void* cu_module{nullptr};   // CUmodule (typed as void* to avoid cuda.h in header)
    void* cu_context{nullptr};  // CUcontext the module was loaded into
    int primary_ctx_device{-1};  // CUdevice whose primary context the module retains, or -1

    ~cuda_module();  // defined in sycl_ptx_kernel.cpp to unload the CUmodule
};

// A kernel function looked up in a module.  It holds the module, so the module stays loaded for as
// long as a pending launch holds the function.
struct cuda_function {
    std::shared_ptr<cuda_module> module;
    void* cu_function{nullptr};    // CUfunction
    int max_threads_per_block{0};  // CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK of cu_function
    // CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES of cu_function: its static .shared variables, which sit at the start of the
    // .shared window, ahead of the dynamic shared memory the __local arguments are placed in.
    int static_shared_bytes{0};
};

// The PTX of one compiled batch and the CUDA modules loaded from it, shared by every entry point of
// the batch and by their clones.  Loading a module JIT-compiles the whole PTX, so each module is
// loaded once for the batch rather than once for every kernel in it.
class ptx_program {
public:
    // One module per launch layout (a block layout combined with a grid layout, see sycl_ptx_kernel.cpp): the
    // layout is baked into the PTX special registers, so each layout a kernel is launched through needs its own
    // rewritten module.
    static constexpr size_t layout_count = 9;

    // The largest alignment an OpenCL type needs (double16, long16), used for a __local parameter whose alignment
    // the PTX does not record: binaries cached before patch_ptx_local kept it.
    static constexpr uint32_t default_local_alignment = 128;

    explicit ptx_program(std::string ptx);

    const std::string& source() const { return _ptx; }

    // Alignment, by parameter index, that each __local pointer parameter of entry point `kernel_name` needs: the
    // N of the '.align N' NVVM declares, which patch_ptx_local keeps as a comment. Entries for other parameters are
    // 0, and the vector is empty for an entry point the PTX does not have.
    const std::vector<uint32_t>& local_param_alignments(const std::string& kernel_name) const;

    // Module for the given launch layout, loaded on first use (inside host_task).  requested_by
    // names the kernel that asked for it in error messages.
    std::shared_ptr<cuda_module> module(size_t layout_idx, const std::string& requested_by);

    // The same module compiled with CU_JIT_THREADS_PER_BLOCK = block_threads (inside host_task).
    std::shared_ptr<cuda_module> capped_module(size_t layout_idx,
                                               unsigned int block_threads,
                                               const std::string& requested_by);

private:
    const std::string _ptx;
    // local_param_alignments of every entry point, read from the PTX once.
    std::map<std::string, std::vector<uint32_t>> _local_alignments;

    std::array<std::once_flag, layout_count> _module_once;
    std::array<std::shared_ptr<cuda_module>, layout_count> _modules;

    // Register-capped modules, keyed by launch layout and block size.
    std::mutex _capped_mutex;
    std::map<std::pair<size_t, unsigned int>, std::shared_ptr<cuda_module>> _capped_modules;
};

// One entry point of a ptx_program and the functions looked up for it.  A launch records a host_task
// that runs after enqueue_kernel() has returned, by which time the caller may have released the
// kernel, so the task holds this state by shared_ptr instead of pointing at the kernel.
struct ptx_kernel_state {
    ptx_kernel_state(std::shared_ptr<ptx_program> program, std::string kernel_name);

    // Function to launch a block of `block_threads` work-items through the given launch layout (called
    // inside host_task).  The JIT sizes a kernel's registers without knowing the work-group size,
    // so a register-heavy kernel can come out unable to launch with a large block: at 133
    // registers, 512 work-items need 68096 of the 65536 an A100 multiprocessor has.  When the
    // block is larger than the plain module allows, the function is taken from a module compiled
    // with CU_JIT_THREADS_PER_BLOCK, which keeps ptxas within the registers that block size leaves.
    std::shared_ptr<cuda_function> function_for_launch(size_t layout_idx, unsigned int block_threads);

    const std::shared_ptr<ptx_program> program;
    const std::string kernel_name;
    // ptx_program::local_param_alignments for this entry point, looked up once rather than at every launch.
    const std::vector<uint32_t> local_alignments;

private:
    std::array<std::once_flag, ptx_program::layout_count> _function_once;
    std::array<std::shared_ptr<cuda_function>, ptx_program::layout_count> _functions;

    // Functions from register-capped modules, keyed by launch layout and block size.
    std::mutex _capped_mutex;
    std::map<std::pair<size_t, unsigned int>, std::shared_ptr<cuda_function>> _capped_functions;
};

// A compiled PTX kernel with stored argument state.
// CUmodule/CUfunction are loaded lazily on first use inside a SYCL host_task.
class sycl_ptx_kernel : public sycl_base_kernel {
public:
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

private:
    std::shared_ptr<ptx_kernel_state> _state;
    std::string _kernel_id;
};

}  // namespace sycl
}  // namespace cldnn
