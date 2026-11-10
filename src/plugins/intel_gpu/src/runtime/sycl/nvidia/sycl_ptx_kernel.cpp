// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_ptx_kernel.hpp"
#include "../sycl_common.hpp"
#include "../sycl_memory.hpp"
#include "openvino/core/except.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <cuda.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace cldnn {
namespace sycl {

namespace {

// CUDA launch limits.
constexpr unsigned int CUDA_MAX_BLOCK_Z = 64;
constexpr unsigned int CUDA_MAX_BLOCK_THREADS = 1024;
constexpr unsigned int CUDA_MAX_GRID_X = 2147483647u;
constexpr unsigned int CUDA_MAX_GRID_YZ = 65535;

// How a launch places the NDRange on the CUDA axes, which cap block.z at 64 and grid.y/z at 65535. See
// "Work-group layout on CUDA" in compat_common.cl for what the kernel sees in each block layout.
enum class block_layout : uint8_t {
    plain,      // dimension d on axis d
    exchanged,  // local size (X, 1, Z) with Z > 64: the block (X, Z, 1)
    folded,     // local size (X, Y, Z) with Z > 64 and Y > 1: the block (X, Z, Y), dimensions 1 and 2 folded
};
constexpr size_t block_layout_count = 3;

enum class grid_layout : uint8_t {
    plain,     // work-group dimension d on axis d
    swap_x_y,  // work-group dimensions 0 and 1 exchanged
    swap_x_z,  // work-group dimensions 0 and 2 exchanged
};
constexpr size_t grid_layout_count = 3;

const char* layout_name(block_layout layout) {
    switch (layout) {
    case block_layout::plain:
        return "plain";
    case block_layout::exchanged:
        return "exchanged";
    case block_layout::folded:
        return "folded";
    }
    return "unknown";
}

const char* layout_name(grid_layout layout) {
    switch (layout) {
    case grid_layout::plain:
        return "plain";
    case grid_layout::swap_x_y:
        return "swap_x_y";
    case grid_layout::swap_x_z:
        return "swap_x_z";
    }
    return "unknown";
}

// One block layout with one grid layout, numbered for the per-layout module caches of ptx_program.
struct launch_layout {
    block_layout block = block_layout::plain;
    grid_layout grid = grid_layout::plain;

    size_t index() const { return static_cast<size_t>(block) * grid_layout_count + static_cast<size_t>(grid); }
    static launch_layout from_index(size_t index) {
        return {static_cast<block_layout>(index / grid_layout_count), static_cast<grid_layout>(index % grid_layout_count)};
    }
    std::string name() const { return std::string("block ") + layout_name(block) + ", grid " + layout_name(grid); }
};
static_assert(block_layout_count * grid_layout_count == ptx_program::layout_count,
              "ptx_program::layout_count has to cover every launch layout");

// The %envreg placeholders the NVIDIA compatibility prelude reads instead of the special registers that depend on the
// block layout, and the register each block layout replaces them with. Keep them in sync with "Work-group layout on
// CUDA" in compat_common.cl.
constexpr size_t layout_placeholder_count = 5;
constexpr const char* layout_placeholders[layout_placeholder_count] = {
    "%envreg27", "%envreg28", "%envreg29", "%envreg30", "%envreg31"};
constexpr const char* layout_placeholder_targets[block_layout_count][layout_placeholder_count] = {
    // local id 1, local id 2, local size 1, local size 2 | fold divisor
    {"%tid.y", "%tid.z", "%ntid.y", "%ntid.z", "%ntid.y"},  // plain
    {"%tid.z", "%tid.y", "%ntid.z", "%ntid.y", "%ntid.y"},  // exchanged
    {"%tid.y", "%tid.z", "%ntid.z", "%ntid.y", "%ntid.z"},  // folded
};

// The PTX to load for a launch layout: the placeholders replaced with the registers of its block layout, the y and z
// extents of .reqntid/.maxntid exchanged when the block layout exchanges those axes, and %ctaid/%nctaid renamed to the
// axes its grid layout puts the work-group dimensions on.
std::string apply_launch_layout(const std::string& ptx, const launch_layout& layout) {
    const auto& targets = layout_placeholder_targets[static_cast<size_t>(layout.block)];

    static constexpr char axis_suffix[3] = {'x', 'y', 'z'};
    // grid_axis[d] is the CUDA grid axis that carries work-group dimension d.
    size_t grid_axis[3] = {0, 1, 2};
    if (layout.grid == grid_layout::swap_x_y)
        std::swap(grid_axis[0], grid_axis[1]);
    else if (layout.grid == grid_layout::swap_x_z)
        std::swap(grid_axis[0], grid_axis[2]);

    std::string out;
    out.reserve(ptx.size());
    for (size_t i = 0; i < ptx.size();) {
        if (ptx[i] != '%') {
            out += ptx[i++];
            continue;
        }
        bool replaced = false;
        for (size_t p = 0; p < layout_placeholder_count && !replaced; ++p) {
            // A placeholder is a whole register name: %envreg27 is not the start of %envreg270.
            const size_t len = std::strlen(layout_placeholders[p]);
            if (ptx.compare(i, len, layout_placeholders[p]) == 0 &&
                (i + len == ptx.size() || !std::isdigit(static_cast<unsigned char>(ptx[i + len])))) {
                out += targets[p];
                i += len;
                replaced = true;
            }
        }
        // The leading '%' keeps the prefixes apart: '%ctaid.' never matches inside '%nctaid.'.
        for (const char* prefix : {"%ctaid.", "%nctaid."}) {
            if (replaced)
                break;
            const size_t len = std::strlen(prefix);
            if (i + len >= ptx.size() || ptx.compare(i, len, prefix) != 0)
                continue;
            const char* suffix = std::find(std::begin(axis_suffix), std::end(axis_suffix), ptx[i + len]);
            if (suffix == std::end(axis_suffix))
                continue;
            out += prefix;
            out += axis_suffix[grid_axis[static_cast<size_t>(suffix - std::begin(axis_suffix))]];
            i += len + 1;
            replaced = true;
        }
        if (!replaced)
            out += ptx[i++];
    }

    if (layout.block != block_layout::plain) {
        // Both directives list the extents in x, y, z order. Only the three-operand spelling is rewritten, which is
        // the one both compilers emit for reqd_work_group_size; the shorter ones leave z at 1, which never needs a
        // layout other than the plain one.
        static const std::regex ntid_re(R"(\.(reqntid|maxntid)\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+))");
        out = std::regex_replace(out, ntid_re, ".$1 $2, $4, $3");
    }
    return out;
}

// Makes `ctx` current on this thread for its lifetime and restores the thread's previous context afterwards.
// Nothing is pushed when `ctx` is null or already current.
class scoped_cuda_context {
public:
    explicit scoped_cuda_context(CUcontext ctx) {
        CUcontext current = nullptr;
        (void)cuCtxGetCurrent(&current);
        _pushed = ctx != nullptr && ctx != current && cuCtxPushCurrent(ctx) == CUDA_SUCCESS;
    }
    ~scoped_cuda_context() {
        if (_pushed) {
            CUcontext popped = nullptr;
            (void)cuCtxPopCurrent(&popped);
        }
    }
    scoped_cuda_context(const scoped_cuda_context&) = delete;
    scoped_cuda_context& operator=(const scoped_cuda_context&) = delete;

private:
    bool _pushed = false;
};

}  // anonymous namespace

cuda_module::~cuda_module() {
    if (cu_module) {
        CUresult r;
        {
            scoped_cuda_context guard(static_cast<CUcontext>(cu_context));
            r = cuModuleUnload(static_cast<CUmodule>(cu_module));
        }
        if (r != CUDA_SUCCESS) {
            const char* err = nullptr;
            cuGetErrorString(r, &err);
            GPU_DEBUG_LOG << "[GPU][NVIDIA] cuModuleUnload failed: " << (err ? err : "unknown") << std::endl;
        }
        cu_module = nullptr;
    }
    if (primary_ctx_device >= 0) {
        (void)cuDevicePrimaryCtxRelease(static_cast<CUdevice>(primary_ctx_device));
        primary_ctx_device = -1;
    }
}

std::shared_ptr<kernel> sycl_ptx_kernel::clone(bool reuse_kernel_handle) const {
    // The PTX and the modules loaded from it are shared either way. Reusing the handle shares the looked-up
    // functions as well; otherwise the clone looks them up again, which is cheap. Arguments stay per kernel.
    if (reuse_kernel_handle) {
        return std::make_shared<sycl_ptx_kernel>(_state, _kernel_id);
    }
    return std::make_shared<sycl_ptx_kernel>(_state->program, _state->kernel_name, _kernel_id);
}

bool sycl_ptx_kernel::is_same(const kernel& other) const {
    // Like the OpenCL kernel handle this compares, the entry point state is shared only by kernels that reuse a
    // handle (clone(true)); kernels built or cloned separately load their own functions even from equal PTX.
    const auto* o = dynamic_cast<const sycl_ptx_kernel*>(&other);
    return o != nullptr && _state == o->_state;
}

std::vector<uint8_t> sycl_ptx_kernel::get_binary() const {
    const auto& ptx = _state->program->source();
    return std::vector<uint8_t>(ptx.begin(), ptx.end());
}

namespace {

// Loads `ptx` as a module.  A non-zero `min_threads` tells the JIT the block size the kernels have to
// launch with (CU_JIT_THREADS_PER_BLOCK), so that ptxas gives each work-item no more registers than
// that many work-items can share.  `requested_by` names the kernel that asked for it in error messages.
std::shared_ptr<cuda_module> load_cuda_module(const std::string& ptx,
                                              unsigned int min_threads,
                                              const std::string& requested_by) {
    CUmodule mod = nullptr;
    CUresult load_res;
    if (min_threads == 0) {
        load_res = cuModuleLoadData(&mod, ptx.c_str());
    } else {
        CUjit_option option = CU_JIT_THREADS_PER_BLOCK;
        void* value = reinterpret_cast<void*>(static_cast<uintptr_t>(min_threads));
        load_res = cuModuleLoadDataEx(&mod, ptx.c_str(), 1, &option, &value);
    }
    if (load_res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(load_res, &err_str);
        OPENVINO_THROW("[GPU][NVIDIA] CUDA error ", static_cast<int>(load_res),
                       min_threads == 0 ? " in cuModuleLoadData" : " in cuModuleLoadDataEx",
                       " for kernel '", requested_by, "': ", err_str ? err_str : "unknown");
    }
    auto module = std::make_shared<cuda_module>();
    module->cu_module = static_cast<void*>(mod);

    // The module belongs to the context current here (the SYCL device's, inside a host_task); it is unloaded from
    // that context. A primary context is retained so it outlives the module even if the SYCL runtime lets go
    // first: unloading from a released primary context fails with CUDA_ERROR_CONTEXT_IS_DESTROYED. Other
    // contexts cannot be retained, so for them the module only switches to them for the unload.
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx != nullptr) {
        module->cu_context = static_cast<void*>(ctx);
        CUdevice dev = 0;
        CUcontext primary = nullptr;
        if (cuCtxGetDevice(&dev) == CUDA_SUCCESS && cuDevicePrimaryCtxRetain(&primary, dev) == CUDA_SUCCESS) {
            if (primary == ctx) {
                module->primary_ctx_device = static_cast<int>(dev);
            } else {
                (void)cuDevicePrimaryCtxRelease(dev);
            }
        }
    }
    return module;
}

// Looks up `kernel_name` in `module`.
std::shared_ptr<cuda_function> get_cuda_function(const std::shared_ptr<cuda_module>& module,
                                                 const std::string& kernel_name) {
    CUfunction func = nullptr;
    CUresult r = cuModuleGetFunction(&func, static_cast<CUmodule>(module->cu_module), kernel_name.c_str());
    if (r != CUDA_SUCCESS) {
        const char* err = nullptr;
        cuGetErrorString(r, &err);
        OPENVINO_THROW("[GPU][NVIDIA] cuModuleGetFunction failed for '", kernel_name,
                       "': ", err ? err : "unknown");
    }

    int max_threads = 0;
    r = cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func);
    if (r != CUDA_SUCCESS) {
        const char* err = nullptr;
        cuGetErrorString(r, &err);
        OPENVINO_THROW("[GPU][NVIDIA] cuFuncGetAttribute(MAX_THREADS_PER_BLOCK) failed for '", kernel_name,
                       "': ", err ? err : "unknown");
    }

    int static_shared_bytes = 0;
    r = cuFuncGetAttribute(&static_shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
    if (r != CUDA_SUCCESS) {
        const char* err = nullptr;
        cuGetErrorString(r, &err);
        OPENVINO_THROW("[GPU][NVIDIA] cuFuncGetAttribute(SHARED_SIZE_BYTES) failed for '", kernel_name,
                       "': ", err ? err : "unknown");
    }

    auto function = std::make_shared<cuda_function>();
    function->module = module;
    function->cu_function = static_cast<void*>(func);
    function->max_threads_per_block = max_threads;
    function->static_shared_bytes = static_shared_bytes;
    return function;
}

}  // anonymous namespace

ptx_program::ptx_program(std::string ptx) : _ptx(std::move(ptx)) {
    // A parameter list holds no ')', so it ends at the first one after '.entry name('; parameters are separated by
    // ',', which neither a declaration nor the comment patch_ptx_local leaves contains.
    static const std::regex entry_re(R"(\.entry\s+([\w.$]+)\s*\(([^)]*)\))");
    static const std::regex local_align_re(R"(/\*\s*\.ptr\s+\.shared\s+\.align\s+(\d+)\s*\*/)");
    for (std::sregex_iterator it(_ptx.begin(), _ptx.end(), entry_re), end; it != end; ++it) {
        auto& alignments = _local_alignments[(*it)[1].str()];
        const std::string params = (*it)[2].str();
        size_t start = 0;
        while (true) {
            const size_t comma = params.find(',', start);
            const std::string param = params.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            std::smatch m;
            alignments.push_back(std::regex_search(param, m, local_align_re)
                                     ? static_cast<uint32_t>(std::stoul(m[1].str()))
                                     : 0u);
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
    }
}

const std::vector<uint32_t>& ptx_program::local_param_alignments(const std::string& kernel_name) const {
    static const std::vector<uint32_t> none;
    const auto it = _local_alignments.find(kernel_name);
    return it != _local_alignments.end() ? it->second : none;
}

std::shared_ptr<cuda_module> ptx_program::module(size_t layout_idx, const std::string& requested_by) {
    OPENVINO_ASSERT(layout_idx < layout_count,
                    "[GPU][NVIDIA] Invalid work-group layout index ", layout_idx, " for kernel '", requested_by, "'");
    std::call_once(_module_once[layout_idx], [&]() {
        _modules[layout_idx] =
            load_cuda_module(apply_launch_layout(_ptx, launch_layout::from_index(layout_idx)), 0, requested_by);
    });
    return _modules[layout_idx];
}

std::shared_ptr<cuda_module> ptx_program::capped_module(size_t layout_idx,
                                                        unsigned int block_threads,
                                                        const std::string& requested_by) {
    std::lock_guard<std::mutex> lock(_capped_mutex);
    auto& capped = _capped_modules[{layout_idx, block_threads}];
    if (!capped) {
        capped = load_cuda_module(apply_launch_layout(_ptx, launch_layout::from_index(layout_idx)),
                                  block_threads,
                                  requested_by);
    }
    return capped;
}

ptx_kernel_state::ptx_kernel_state(std::shared_ptr<ptx_program> program, std::string kernel_name)
    : program(std::move(program))
    , kernel_name(std::move(kernel_name))
    , local_alignments(this->program->local_param_alignments(this->kernel_name)) {}

std::shared_ptr<cuda_function> ptx_kernel_state::function_for_launch(size_t layout_idx, unsigned int block_threads) {
    OPENVINO_ASSERT(layout_idx < ptx_program::layout_count,
                    "[GPU][NVIDIA] Invalid work-group layout index ", layout_idx, " for kernel '", kernel_name, "'");
    std::call_once(_function_once[layout_idx], [&]() {
        _functions[layout_idx] = get_cuda_function(program->module(layout_idx, kernel_name), kernel_name);
    });
    const auto& plain = _functions[layout_idx];
    if (block_threads <= static_cast<unsigned int>(plain->max_threads_per_block))
        return plain;

    // Lock order is always this state, then the program: the program never calls back into a kernel's state.
    std::lock_guard<std::mutex> lock(_capped_mutex);
    auto& capped = _capped_functions[{layout_idx, block_threads}];
    if (!capped) {
        capped = get_cuda_function(program->capped_module(layout_idx, block_threads, kernel_name), kernel_name);
    }

    if (block_threads > static_cast<unsigned int>(capped->max_threads_per_block)) {
        OPENVINO_THROW("[GPU][NVIDIA] Kernel '", kernel_name, "' cannot be launched with ", block_threads,
                       " work-items per work-group: the CUDA function allows at most ",
                       plain->max_threads_per_block, ", and ", capped->max_threads_per_block,
                       " when compiled for that block size");
    }
    return capped;
}

void sycl_ptx_kernel::launch(::sycl::handler& cgh,
                             const kernel_arguments_desc& args_desc) {
    auto args = stored_args();
    const auto& gws = args_desc.workGroups.global;
    const auto& lws = args_desc.workGroups.local;

    // Work-group size and count per NDRange dimension. They stay in size_t until they are checked against the CUDA
    // limits: narrowing first would wrap an extent of 2^32 + 1 to 1 and run only part of the NDRange. A dimension
    // without a local size runs one work-item per block.
    size_t block_extent[3];
    size_t grid_extent[3];
    for (size_t d = 0; d < 3; ++d) {
        const size_t global = d < gws.size() ? gws[d] : 1;
        const size_t local = (d < lws.size() && lws[d] > 0) ? lws[d] : 1;
        block_extent[d] = local;
        grid_extent[d] = global / local + (global % local != 0 ? 1 : 0);
    }

    // CUDA caps the axes asymmetrically (block.z <= 64, grid.y and grid.z <= 65535), so the launch picks how the
    // NDRange sits on them; see "Work-group layout on CUDA" in compat_common.cl for what each choice means to the
    // kernel. Splitting a work-group across blocks instead would break barriers and __local sharing.
    //  - Block: a local size (X, Y, Z) with Z > 64 runs as the block (X, Z, Y), dimensions 1 and 2 folded so that the
    //    linear local id stays the OpenCL one. With Y == 1 this is a plain exchange of the two axes.
    //  - Grid: a work-group count above 65535 along dimension 1 or 2 is moved to x. Work-groups are independent, so
    //    their order does not matter and any exchange will do.
    launch_layout layout;
    if (block_extent[2] > CUDA_MAX_BLOCK_Z)
        layout.block = block_extent[1] == 1 ? block_layout::exchanged : block_layout::folded;
    if (grid_extent[1] > CUDA_MAX_GRID_YZ)
        layout.grid = grid_layout::swap_x_y;
    else if (grid_extent[2] > CUDA_MAX_GRID_YZ)
        layout.grid = grid_layout::swap_x_z;

    // Within 1024 work-items, every block layout keeps each axis within its limit: dimension 0 stays on x, and a fold
    // puts Z <= 1024 on y and Y on z, where Y * Z <= 1024 and Z > 64 leave Y <= 15. The extents are checked one by
    // one first, so that the product cannot overflow.
    if (block_extent[0] > CUDA_MAX_BLOCK_THREADS || block_extent[1] > CUDA_MAX_BLOCK_THREADS ||
        block_extent[2] > CUDA_MAX_BLOCK_THREADS ||
        block_extent[0] * block_extent[1] * block_extent[2] > CUDA_MAX_BLOCK_THREADS) {
        OPENVINO_THROW("[GPU][NVIDIA] Kernel '", _state->kernel_name, "' cannot be launched: its work-group (",
                       block_extent[0], ", ", block_extent[1], ", ", block_extent[2], ") has more than ",
                       CUDA_MAX_BLOCK_THREADS, " work-items, the most a CUDA block can hold");
    }

    size_t block_axes[3] = {block_extent[0], block_extent[1], block_extent[2]};
    if (layout.block != block_layout::plain)
        std::swap(block_axes[1], block_axes[2]);
    size_t grid_axes[3] = {grid_extent[0], grid_extent[1], grid_extent[2]};
    if (layout.grid == grid_layout::swap_x_y)
        std::swap(grid_axes[0], grid_axes[1]);
    else if (layout.grid == grid_layout::swap_x_z)
        std::swap(grid_axes[0], grid_axes[2]);

    if (grid_axes[0] > CUDA_MAX_GRID_X || grid_axes[1] > CUDA_MAX_GRID_YZ || grid_axes[2] > CUDA_MAX_GRID_YZ) {
        OPENVINO_THROW("[GPU][NVIDIA] Kernel '", _state->kernel_name, "' cannot be launched: its work-group count (",
                       grid_extent[0], ", ", grid_extent[1], ", ", grid_extent[2], ") does not fit a CUDA grid, ",
                       "which takes more than ", CUDA_MAX_GRID_YZ, " work-groups along one dimension only, and at most ",
                       CUDA_MAX_GRID_X, " along that one");
    }

    // Every extent is within unsigned int here.
    const unsigned int bx = static_cast<unsigned int>(block_axes[0]);
    const unsigned int by = static_cast<unsigned int>(block_axes[1]);
    const unsigned int bz = static_cast<unsigned int>(block_axes[2]);
    const unsigned int gx = static_cast<unsigned int>(grid_axes[0]);
    const unsigned int gy = static_cast<unsigned int>(grid_axes[1]);
    const unsigned int gz = static_cast<unsigned int>(grid_axes[2]);

    // Categorize args: gpu_buffer (need SYCL accessors) vs gpu_usm (direct pointer)
    std::vector<size_t> buf_arg_indices;
    std::vector<size_t> usm_arg_indices;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i].kind == sycl_ptx_kernel::arg_t::kind_t::BUFFER && args[i].mem) {
            if (dynamic_cast<const gpu_buffer*>(args[i].mem.get())) {
                buf_arg_indices.push_back(i);
            } else {
                usm_arg_indices.push_back(i);
            }
        }
    }

    // Build gpu_buffer pointer list for accessor creation inside submit handler
    std::vector<std::pair<const gpu_buffer*, size_t>> buf_ptrs;
    for (auto idx : buf_arg_indices) {
        auto* gb = dynamic_cast<const gpu_buffer*>(args[idx].mem.get());
        buf_ptrs.emplace_back(gb, gb->get_byte_offset());
    }

    // Collect USM device pointers (no accessors needed)
    std::vector<std::pair<size_t, void*>> usm_ptrs;
    for (auto idx : usm_arg_indices) {
        void* ptr = args[idx].mem->buffer_ptr();
        usm_ptrs.emplace_back(idx, ptr);
    }

    struct buf_info_t {
        ::sycl::accessor<std::byte, 1, ::sycl::access::mode::read_write> acc;
        size_t byte_offset;
    };

    // The owning stream has already wired the dependency events onto this
    // command group (handler) and will submit it and create the resulting
    // event.  Here we only record the backend-specific dispatch.
    std::vector<buf_info_t> buf_infos;
    buf_infos.reserve(buf_ptrs.size());
    for (auto& [gb, off] : buf_ptrs) {
        buf_infos.push_back({gb->create_rw_accessor(cgh), off});
    }

    // The task runs once the dependencies complete, possibly after the caller has released this
    // kernel, so it holds the entry point state it needs rather than a pointer to the kernel.
    cldnn::sycl::host_task(cgh, [buf_infos = std::move(buf_infos),
                                 args = std::move(args),
                                 buf_arg_indices,
                                 usm_ptrs,
                                 state = _state,
                                 layout,
                                 gx, gy, gz, bx, by, bz](::sycl::interop_handle ih) {
        const auto function = state->function_for_launch(layout.index(), bx * by * bz);
        auto cu_stream = ih.get_native_queue<::sycl::backend::ext_oneapi_cuda>();
        auto cu_func = static_cast<CUfunction>(function->cu_function);

        std::vector<std::vector<uint8_t>> param_storage;
        std::vector<void*> params;

        // __local pointer arguments are .shared addresses (.param .u64 .ptr .shared in NVVM's PTX). The static
        // .shared variables of the function - __local arrays declared in the kernel, the work-group collectives'
        // buffer of compat_wg.cl - take the start of the .shared window and the dynamic shared memory follows them,
        // so the arguments are placed from the end of the static ones, each at the alignment it needs.
        uint64_t smem_offset = static_cast<uint64_t>(function->static_shared_bytes);

        for (size_t i = 0; i < args.size(); ++i) {
            const auto& arg = args[i];
            if (arg.kind == sycl_ptx_kernel::arg_t::kind_t::BUFFER) {
                auto buf_it = std::find(buf_arg_indices.begin(), buf_arg_indices.end(), i);
                if (buf_it != buf_arg_indices.end() && arg.mem) {
                    size_t local_idx = static_cast<size_t>(buf_it - buf_arg_indices.begin());
                    const auto& bi = buf_infos[local_idx];
                    auto raw_ptr = ih.get_native_mem<::sycl::backend::ext_oneapi_cuda>(bi.acc);
                    CUdeviceptr cu_ptr =
                        reinterpret_cast<CUdeviceptr>(raw_ptr) + bi.byte_offset;
                    param_storage.emplace_back(sizeof(CUdeviceptr));
                    std::memcpy(param_storage.back().data(), &cu_ptr, sizeof(CUdeviceptr));
                    params.push_back(param_storage.back().data());
                } else {
                    auto usm_it = std::find_if(usm_ptrs.begin(), usm_ptrs.end(),
                        [i](const std::pair<size_t, void*>& p) { return p.first == i; });
                    if (usm_it != usm_ptrs.end()) {
                        CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(usm_it->second);
                        param_storage.emplace_back(sizeof(CUdeviceptr));
                        std::memcpy(param_storage.back().data(), &cu_ptr, sizeof(CUdeviceptr));
                        params.push_back(param_storage.back().data());
                    } else {
                        CUdeviceptr cu_ptr = 0;
                        param_storage.emplace_back(sizeof(CUdeviceptr));
                        std::memcpy(param_storage.back().data(), &cu_ptr, sizeof(CUdeviceptr));
                        params.push_back(param_storage.back().data());
                    }
                }
            } else if (arg.kind == sycl_ptx_kernel::arg_t::kind_t::SCALAR) {
                size_t sz = 4;
                switch (arg.scalar.t) {
                    case scalar_desc::Types::UINT8:   sz = 1; break;
                    case scalar_desc::Types::INT8:    sz = 1; break;
                    case scalar_desc::Types::UINT16:  sz = 2; break;
                    case scalar_desc::Types::INT16:   sz = 2; break;
                    case scalar_desc::Types::UINT32:  sz = 4; break;
                    case scalar_desc::Types::INT32:   sz = 4; break;
                    case scalar_desc::Types::FLOAT32: sz = 4; break;
                    case scalar_desc::Types::UINT64:  sz = 8; break;
                    case scalar_desc::Types::INT64:   sz = 8; break;
                    case scalar_desc::Types::FLOAT64: sz = 8; break;
                }
                param_storage.emplace_back(sz);
                std::memcpy(param_storage.back().data(), &arg.scalar.v, sz);
                params.push_back(param_storage.back().data());
            } else if (arg.kind == sycl_ptx_kernel::arg_t::kind_t::LOCAL_MEM) {
                // The parameter value is the argument's address in the .shared window, rounded up to the
                // alignment the kernel was compiled to assume for it.
                const uint64_t align = (i < state->local_alignments.size() && state->local_alignments[i] > 0)
                                           ? state->local_alignments[i]
                                           : ptx_program::default_local_alignment;
                smem_offset = (smem_offset + align - 1) / align * align;
                param_storage.emplace_back(sizeof(uint64_t));
                std::memcpy(param_storage.back().data(), &smem_offset, sizeof(uint64_t));
                params.push_back(param_storage.back().data());
                smem_offset += static_cast<uint64_t>(arg.local_size);
            }
        }

        // The dynamic shared memory to request: everything past the static variables up to the end of the last
        // __local argument, alignment padding included.
        const unsigned int shared_mem =
            static_cast<unsigned int>(smem_offset - static_cast<uint64_t>(function->static_shared_bytes));

        // The launch shape, for error messages: a kernel is launched with many shapes, and a failure is only
        // reproducible with the one that failed.
        const auto launch_shape = [&]() {
            return "grid (" + std::to_string(gx) + ", " + std::to_string(gy) + ", " + std::to_string(gz) +
                   "), block (" + std::to_string(bx) + ", " + std::to_string(by) + ", " + std::to_string(bz) +
                   "), " + std::to_string(shared_mem) + " bytes of shared memory, " + layout.name();
        };

        CUresult res = cuLaunchKernel(cu_func,
                                      gx, gy, gz,
                                      bx, by, bz,
                                      shared_mem,
                                      cu_stream,
                                      params.empty() ? nullptr : params.data(),
                                      nullptr);
        if (res != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(res, &err_str);
            OPENVINO_THROW("[GPU][NVIDIA] cuLaunchKernel failed for '", state->kernel_name, "' with ",
                           launch_shape(), ": ", err_str ? err_str : "unknown error");
        }

#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
        // An ordinary host_task's event completes when this callback returns, so without the native-command
        // extension the stream is synchronised to keep the event from completing while the kernel still accesses
        // its buffers. A native command's event covers the work it enqueues, so that path needs no wait.
        CUresult sync_res = cuStreamSynchronize(cu_stream);
        if (sync_res != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(sync_res, &err_str);
            OPENVINO_THROW("[GPU][NVIDIA] cuStreamSynchronize failed for '", state->kernel_name, "' with ",
                           launch_shape(), ": ", err_str ? err_str : "unknown error");
        }
#endif
    });
}

}  // namespace sycl
}  // namespace cldnn
