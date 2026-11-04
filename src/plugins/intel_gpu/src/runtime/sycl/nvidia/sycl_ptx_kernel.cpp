// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_ptx_kernel.hpp"
#include "../sycl_common.hpp"
#include "../sycl_memory.hpp"
#include "openvino/core/except.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        CUresult _r = (call);                                                            \
        if (_r != CUDA_SUCCESS) {                                                        \
            const char* _err_str = nullptr;                                              \
            cuGetErrorString(_r, &_err_str);                                             \
            OPENVINO_THROW("[GPU][NVIDIA] CUDA error ", static_cast<int>(_r),            \
                           " in " #call ": ", _err_str ? _err_str : "unknown");          \
        }                                                                                \
    } while (0)

namespace cldnn {
namespace sycl {

namespace {

bool is_env_enabled(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0' && v[0] != '0';
}

bool cuda_launch_trace_enabled() {
    static const bool enabled = is_env_enabled("OV_SYCL_CUDA_LAUNCH_TRACE");
    return enabled;
}

bool cuda_launch_sync_enabled() {
    static const bool enabled = is_env_enabled("OV_SYCL_CUDA_LAUNCH_SYNC");
    return enabled;
}

std::atomic<uint64_t>& cuda_launch_counter() {
    static std::atomic<uint64_t> counter{0};
    return counter;
}

void print_launch_trace(uint64_t launch_id,
                        const std::string& kernel_name,
                        unsigned int gx,
                        unsigned int gy,
                        unsigned int gz,
                        unsigned int bx,
                        unsigned int by,
                        unsigned int bz,
                        unsigned int shared_mem,
                        size_t num_args,
                        size_t num_buffer_args,
                        size_t num_scalar_args,
                        size_t num_local_args,
                        size_t mapping_idx,
                        const char* phase,
                        const char* err) {
    std::fprintf(stderr,
                 "[OV_SYCL][CUDA_LAUNCH][%s] id=%llu kernel=%s grid=(%u,%u,%u) block=(%u,%u,%u) "
                 "shared=%u args=%zu buffers=%zu scalars=%zu local=%zu dim_map=%zu err=%s\n",
                 phase,
                 static_cast<unsigned long long>(launch_id),
                 kernel_name.c_str(),
                 gx,
                 gy,
                 gz,
                 bx,
                 by,
                 bz,
                 shared_mem,
                 num_args,
                 num_buffer_args,
                 num_scalar_args,
                 num_local_args,
                 mapping_idx,
                 err ? err : "none");
    std::fflush(stderr);
}

// CUDA launch limits.  The axes are capped asymmetrically, which is what makes the
// NDRange-to-axis assignment matter: an extent of 192 is fine on x or y but not on z.
constexpr unsigned int CUDA_MAX_BLOCK_XY = 1024;
constexpr unsigned int CUDA_MAX_BLOCK_Z = 64;
constexpr unsigned int CUDA_MAX_BLOCK_THREADS = 1024;
constexpr unsigned int CUDA_MAX_GRID_X = 2147483647u;
constexpr unsigned int CUDA_MAX_GRID_YZ = 65535;

// Distributes the per-dimension extents over the CUDA axes according to `mapping`.
void apply_dim_mapping(const unsigned int (&per_dim)[3], const dim_mapping& mapping, unsigned int (&per_axis)[3]) {
    per_axis[0] = per_axis[1] = per_axis[2] = 1;
    for (size_t d = 0; d < 3; ++d)
        per_axis[mapping[d]] = per_dim[d];
}

// The NVIDIA sub-group emulation builds its warp member masks from the hardware %laneid but takes
// the sub-group local id from the OpenCL linear local id (see compat_sg.cl).  The two only
// describe the same lanes while the work-group dimensions that actually span more than one
// work-item keep their relative order across the CUDA axes, which the hardware enumerates
// x-fastest.  Dimensions of extent 1 contribute to neither linear id, so they are free to move.
bool preserves_local_id_order(const unsigned int (&block_per_dim)[3], const dim_mapping& mapping) {
    int previous_axis = -1;
    for (size_t d = 0; d < 3; ++d) {
        if (block_per_dim[d] <= 1)
            continue;
        if (static_cast<int>(mapping[d]) <= previous_axis)
            return false;
        previous_axis = static_cast<int>(mapping[d]);
    }
    return true;
}

bool dims_fit_cuda_limits(const unsigned int (&block_per_dim)[3],
                          const unsigned int (&grid_per_dim)[3],
                          const dim_mapping& mapping) {
    unsigned int block[3];
    unsigned int grid[3];
    apply_dim_mapping(block_per_dim, mapping, block);
    apply_dim_mapping(grid_per_dim, mapping, grid);

    // The thread count per block is the same for every mapping; only the per-axis caps
    // tell the mappings apart.  It is still checked here so that a work-group that is too
    // large outright matches no mapping and is rejected at launch.
    if (static_cast<uint64_t>(block[0]) * block[1] * block[2] > CUDA_MAX_BLOCK_THREADS)
        return false;

    return block[0] <= CUDA_MAX_BLOCK_XY && block[1] <= CUDA_MAX_BLOCK_XY && block[2] <= CUDA_MAX_BLOCK_Z &&
           grid[0] <= CUDA_MAX_GRID_X && grid[1] <= CUDA_MAX_GRID_YZ && grid[2] <= CUDA_MAX_GRID_YZ;
}

// NVVM and the LLVM NVPTX backend read work-item coordinates from %tid/%ntid and work-group
// coordinates from %ctaid/%nctaid, the suffix naming the axis.  Carrying an NDRange dimension
// on a different axis therefore means rewriting those suffixes, plus the .reqntid/.maxntid
// directives that state the block shape the driver validates the launch against.
std::string remap_ptx_dims(const std::string& ptx, const dim_mapping& mapping) {
    if (mapping[0] == 0 && mapping[1] == 1 && mapping[2] == 2)
        return ptx;

    static constexpr char axis_suffix[3] = {'x', 'y', 'z'};
    static const char* const reg_prefixes[] = {"%tid.", "%ntid.", "%ctaid.", "%nctaid."};

    std::string remapped;
    remapped.reserve(ptx.size());
    for (size_t i = 0; i < ptx.size();) {
        bool rewritten = false;
        // The leading '%' keeps the prefixes from overlapping: '%ntid.' never matches
        // '%tid.', and '%nctaid.' never matches '%ctaid.'.
        if (ptx[i] == '%') {
            for (const char* prefix : reg_prefixes) {
                const size_t len = std::strlen(prefix);
                if (i + len >= ptx.size() || ptx.compare(i, len, prefix) != 0)
                    continue;
                const char* suffix = std::find(std::begin(axis_suffix), std::end(axis_suffix), ptx[i + len]);
                if (suffix == std::end(axis_suffix))
                    continue;
                remapped += prefix;
                remapped += axis_suffix[mapping[static_cast<size_t>(suffix - std::begin(axis_suffix))]];
                i += len + 1;
                rewritten = true;
                break;
            }
        }
        if (!rewritten)
            remapped += ptx[i++];
    }

    // Both directives list the extents in x, y, z order.  Only the three-operand spelling is
    // rewritten, which is the one both compilers emit for reqd_work_group_size.
    static const std::regex ntid_re(R"(\.(reqntid|maxntid)\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+))");
    std::string out;
    out.reserve(remapped.size());
    size_t last = 0;
    for (std::sregex_iterator it(remapped.begin(), remapped.end(), ntid_re), end; it != end; ++it) {
        const size_t pos = static_cast<size_t>(it->position());
        out.append(remapped, last, pos - last);
        const std::string per_dim[3] = {(*it)[2].str(), (*it)[3].str(), (*it)[4].str()};
        std::string per_axis[3] = {"1", "1", "1"};
        for (size_t d = 0; d < 3; ++d)
            per_axis[mapping[d]] = per_dim[d];
        out += "." + (*it)[1].str() + " " + per_axis[0] + ", " + per_axis[1] + ", " + per_axis[2];
        last = pos + static_cast<size_t>(it->length());
    }
    out.append(remapped, last, std::string::npos);
    return out;
}

}  // anonymous namespace

// Ordered by how far each one moves the dimensions: the identity so that anything already
// launchable keeps its current behaviour exactly, then the three single exchanges, then the two
// rotations.  The launch path takes the first entry that both preserves the linear local id order
// and fits the CUDA limits, so a work-group is disturbed as little as the hardware allows.
const std::array<dim_mapping, 6>& dim_mappings() {
    static const std::array<dim_mapping, 6> table = {{
        {{0, 1, 2}},  // identity
        {{0, 2, 1}},  // dimensions 1 and 2 exchanged
        {{2, 1, 0}},  // dimensions 0 and 2 exchanged
        {{1, 0, 2}},  // dimensions 0 and 1 exchanged
        {{1, 2, 0}},  // rotated left
        {{2, 0, 1}},  // rotated right
    }};
    return table;
}

cuda_module_state::~cuda_module_state() {
    if (cu_module) {
        (void)cuModuleUnload(static_cast<CUmodule>(cu_module));
        cu_module = nullptr;
        cu_function = nullptr;
    }
}

std::shared_ptr<kernel> sycl_ptx_kernel::clone(bool /*reuse_kernel_handle*/) const {
    return std::make_shared<sycl_ptx_kernel>(_program->ptx_source, _program->kernel_name, _kernel_id);
}

bool sycl_ptx_kernel::is_same(const kernel& other) const {
    const auto* o = dynamic_cast<const sycl_ptx_kernel*>(&other);
    if (!o) return false;
    return _program->ptx_source == o->_program->ptx_source && _program->kernel_name == o->_program->kernel_name;
}

std::vector<uint8_t> sycl_ptx_kernel::get_binary() const {
    return std::vector<uint8_t>(_program->ptx_source.begin(), _program->ptx_source.end());
}

namespace {

// Loads `ptx` into `state` and looks up `kernel_name` in it.  A non-zero `min_threads` tells the JIT
// the block size the kernel has to launch with (CU_JIT_THREADS_PER_BLOCK), so that ptxas gives each
// work-item no more registers than that many work-items can share.
void load_module(cuda_module_state& state,
                 const std::string& ptx,
                 const std::string& kernel_name,
                 unsigned int min_threads) {
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
                       " for kernel '", kernel_name, "': ", err_str ? err_str : "unknown");
    }

    CUfunction func = nullptr;
    CUresult r = cuModuleGetFunction(&func, mod, kernel_name.c_str());
    if (r != CUDA_SUCCESS) {
        const char* err = nullptr;
        cuGetErrorString(r, &err);
        // Unload module to avoid leaking on failure
        cuModuleUnload(mod);
        OPENVINO_THROW("[GPU][NVIDIA] cuModuleGetFunction failed for '", kernel_name,
                       "': ", err ? err : "unknown");
    }

    int max_threads = 0;
    r = cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func);
    if (r != CUDA_SUCCESS) {
        const char* err = nullptr;
        cuGetErrorString(r, &err);
        cuModuleUnload(mod);
        OPENVINO_THROW("[GPU][NVIDIA] cuFuncGetAttribute(MAX_THREADS_PER_BLOCK) failed for '", kernel_name,
                       "': ", err ? err : "unknown");
    }

    state.cu_module = static_cast<void*>(mod);
    state.cu_function = static_cast<void*>(func);
    state.max_threads_per_block = max_threads;
}

int function_attribute(const cuda_module_state& state, CUfunction_attribute attribute) {
    int value = -1;
    (void)cuFuncGetAttribute(&value, attribute, static_cast<CUfunction>(state.cu_function));
    return value;
}

}  // anonymous namespace

ptx_program_state::ptx_program_state(std::string ptx, std::string name)
    : ptx_source(std::move(ptx))
    , kernel_name(std::move(name)) {
    for (auto& state : module_state)
        state = std::make_shared<cuda_module_state>();
}

void ptx_program_state::ensure_loaded(size_t mapping_idx) {
    OPENVINO_ASSERT(mapping_idx < mapping_count,
                    "[GPU][NVIDIA] Invalid dimension mapping index ", mapping_idx, " for kernel '", kernel_name, "'");
    std::call_once(module_once[mapping_idx], [this, mapping_idx]() {
        load_module(*module_state[mapping_idx],
                    remap_ptx_dims(ptx_source, dim_mappings()[mapping_idx]),
                    kernel_name,
                    0);
    });
}

std::shared_ptr<cuda_module_state> ptx_program_state::module_for_launch(size_t mapping_idx,
                                                                        unsigned int block_threads) {
    ensure_loaded(mapping_idx);
    const auto& plain = module_state[mapping_idx];
    if (block_threads <= static_cast<unsigned int>(plain->max_threads_per_block))
        return plain;

    std::lock_guard<std::mutex> lock(capped_mutex);
    auto& capped = capped_modules[{mapping_idx, block_threads}];
    if (!capped) {
        auto state = std::make_shared<cuda_module_state>();
        load_module(*state, remap_ptx_dims(ptx_source, dim_mappings()[mapping_idx]), kernel_name, block_threads);
        if (cuda_launch_trace_enabled()) {
            std::fprintf(stderr,
                         "[OV_SYCL][CUDA_LAUNCH][capped_module] kernel=%s block_threads=%u dim_map=%zu "
                         "max_threads=%d->%d regs=%d->%d local_bytes=%d->%d\n",
                         kernel_name.c_str(),
                         block_threads,
                         mapping_idx,
                         plain->max_threads_per_block,
                         state->max_threads_per_block,
                         function_attribute(*plain, CU_FUNC_ATTRIBUTE_NUM_REGS),
                         function_attribute(*state, CU_FUNC_ATTRIBUTE_NUM_REGS),
                         function_attribute(*plain, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
                         function_attribute(*state, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES));
            std::fflush(stderr);
        }
        capped = std::move(state);
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

    // Compute grid dimensions (ceiling division)
    auto ceil_div = [](size_t a, size_t b) -> unsigned int {
        return b ? static_cast<unsigned int>((a + b - 1) / b) : 1u;
    };
    unsigned int gx = (lws.size() > 0 && lws[0] > 0) ? ceil_div(gws.size() > 0 ? gws[0] : 1, lws[0]) : static_cast<unsigned int>(gws.size() > 0 ? gws[0] : 1u);
    unsigned int gy = (lws.size() > 1 && lws[1] > 0) ? ceil_div(gws.size() > 1 ? gws[1] : 1, lws[1]) : static_cast<unsigned int>(gws.size() > 1 ? gws[1] : 1u);
    unsigned int gz = (lws.size() > 2 && lws[2] > 0) ? ceil_div(gws.size() > 2 ? gws[2] : 1, lws[2]) : static_cast<unsigned int>(gws.size() > 2 ? gws[2] : 1u);
    unsigned int bx = (lws.size() > 0 && lws[0] > 0) ? static_cast<unsigned int>(lws[0]) : 1u;
    unsigned int by = (lws.size() > 1 && lws[1] > 0) ? static_cast<unsigned int>(lws[1]) : 1u;
    unsigned int bz = (lws.size() > 2 && lws[2] > 0) ? static_cast<unsigned int>(lws[2]) : 1u;

    // CUDA caps the three axes asymmetrically (block.z <= 64, grid.y and grid.z <= 65535),
    // while OpenCL work-groups routinely put the only non-unit extent last, e.g. lws = {1, 1, 192}.
    // Splitting such a work-group across blocks would break barriers and __local sharing, and a
    // kernel built from reqd_work_group_size carries a .reqntid the driver rejects the launch
    // over.  Carrying the NDRange dimensions on different axes keeps the work-group whole; the
    // PTX of the selected mapping is rewritten to read the matching special registers.
    const unsigned int block_per_dim[3] = {bx, by, bz};
    const unsigned int grid_per_dim[3] = {gx, gy, gz};

    size_t mapping_idx = 0;
    bool mapping_found = false;
    for (size_t i = 0; i < dim_mapping_count; ++i) {
        const dim_mapping& candidate = dim_mappings()[i];
        if (preserves_local_id_order(block_per_dim, candidate) &&
            dims_fit_cuda_limits(block_per_dim, grid_per_dim, candidate)) {
            mapping_idx = i;
            mapping_found = true;
            break;
        }
    }

    if (!mapping_found) {
        // Resizing the work-group to fit would change get_local_size/get_group_id and split
        // barriers and __local sharing, so an unmappable launch is rejected instead.
        OPENVINO_THROW("[GPU][NVIDIA] Kernel '", _program->kernel_name, "' cannot be launched: work-group (",
                       block_per_dim[0], ", ", block_per_dim[1], ", ", block_per_dim[2],
                       ") with work-group count (",
                       grid_per_dim[0], ", ", grid_per_dim[1], ", ", grid_per_dim[2],
                       ") does not fit the CUDA launch limits (block <= 1024 threads, block.x/y <= 1024, "
                       "block.z <= 64, grid.y/z <= 65535) under any dimension mapping that keeps the "
                       "local id order");
    }

    unsigned int block[3];
    unsigned int grid[3];
    apply_dim_mapping(block_per_dim, dim_mappings()[mapping_idx], block);
    apply_dim_mapping(grid_per_dim, dim_mappings()[mapping_idx], grid);
    bx = block[0];
    by = block[1];
    bz = block[2];
    gx = grid[0];
    gy = grid[1];
    gz = grid[2];

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
    // kernel, so it holds the PTX and module state it needs rather than a pointer to the kernel.
    cldnn::sycl::host_task(cgh, [buf_infos = std::move(buf_infos),
                                 args = std::move(args),
                                 buf_arg_indices,
                                 usm_ptrs,
                                 program = _program,
                                 mapping_idx,
                                 gx, gy, gz, bx, by, bz](::sycl::interop_handle ih) {
        const auto module_state = program->module_for_launch(mapping_idx, bx * by * bz);
        auto cu_stream = ih.get_native_queue<::sycl::backend::ext_oneapi_cuda>();
        auto cu_func = static_cast<CUfunction>(module_state->cu_function);

        std::vector<std::vector<uint8_t>> param_storage;
        std::vector<void*> params;

        // First pass: accumulate total shared memory so we can compute offsets in order.
        // LOCAL_MEM args map to .param .u64 .ptr .shared in NVIDIA OCL-compiled PTX.
        // The parameter value is the byte offset from the start of dynamic shared memory
        // (address 0 in the .shared address space, no static smem in OCL kernels).
        uint64_t smem_offset = 0;

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
                        if (cuda_launch_trace_enabled()) {
                            std::fprintf(stderr,
                                         "[OV_SYCL][CUDA_LAUNCH][null_buffer] kernel=%s arg_index=%zu\n",
                                         program->kernel_name.c_str(),
                                         i);
                            std::fflush(stderr);
                        }
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
                // NVIDIA OCL-compiled PTX uses .param .u64 .ptr .shared for __local args.
                // Pass the cumulative byte offset within dynamic shared memory as the
                // parameter value (address in .shared address space, starting at 0).
                param_storage.emplace_back(sizeof(uint64_t));
                std::memcpy(param_storage.back().data(), &smem_offset, sizeof(uint64_t));
                params.push_back(param_storage.back().data());
                smem_offset += static_cast<uint64_t>(arg.local_size);
            }
        }

        unsigned int shared_mem = 0;
        size_t num_buffer_args = 0;
        size_t num_scalar_args = 0;
        size_t num_local_args = 0;
        for (const auto& arg : args) {
            if (arg.kind == sycl_ptx_kernel::arg_t::kind_t::BUFFER) {
                ++num_buffer_args;
            } else if (arg.kind == sycl_ptx_kernel::arg_t::kind_t::SCALAR) {
                ++num_scalar_args;
            } else if (arg.kind == sycl_ptx_kernel::arg_t::kind_t::LOCAL_MEM) {
                ++num_local_args;
                shared_mem += static_cast<unsigned int>(arg.local_size);
            }
        }

        const uint64_t launch_id = ++cuda_launch_counter();
        if (cuda_launch_trace_enabled()) {
            print_launch_trace(launch_id,
                               program->kernel_name,
                               gx, gy, gz, bx, by, bz,
                               shared_mem,
                               args.size(),
                               num_buffer_args,
                               num_scalar_args,
                               num_local_args,
                               mapping_idx,
                               "begin",
                               nullptr);
        }

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
            print_launch_trace(launch_id,
                               program->kernel_name,
                               gx, gy, gz, bx, by, bz,
                               shared_mem,
                               args.size(),
                               num_buffer_args,
                               num_scalar_args,
                               num_local_args,
                               mapping_idx,
                               "cuLaunchKernel_fail",
                               err_str);
            OPENVINO_THROW("[GPU][NVIDIA] cuLaunchKernel failed for '",
                           program->kernel_name, "': ",
                           err_str ? err_str : "unknown error");
        }

        if (cuda_launch_sync_enabled()) {
            CUresult sync_res = cuStreamSynchronize(cu_stream);
            if (sync_res != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(sync_res, &err_str);
                print_launch_trace(launch_id,
                                   program->kernel_name,
                                   gx, gy, gz, bx, by, bz,
                                   shared_mem,
                                   args.size(),
                                   num_buffer_args,
                                   num_scalar_args,
                                   num_local_args,
                                   mapping_idx,
                                   "cuStreamSynchronize_fail",
                                   err_str);
                OPENVINO_THROW("[GPU][NVIDIA] cuStreamSynchronize failed for '",
                               program->kernel_name, "': ",
                               err_str ? err_str : "unknown error");
            }
            if (cuda_launch_trace_enabled()) {
                print_launch_trace(launch_id,
                                   program->kernel_name,
                                   gx, gy, gz, bx, by, bz,
                                   shared_mem,
                                   args.size(),
                                   num_buffer_args,
                                   num_scalar_args,
                                   num_local_args,
                                   mapping_idx,
                                   "sync_ok",
                                   nullptr);
            }
        }
    });
}

}  // namespace sycl
}  // namespace cldnn
