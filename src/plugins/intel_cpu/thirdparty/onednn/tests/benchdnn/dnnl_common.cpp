/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#include <algorithm> // for std::reverse and std::copy
#include <cctype> // for std::isdigit
#include <functional> // for std::bind and std::placeholders
#include <string> // for std::string
#include <utility> // for std::pair
#include <vector> // for std::vector

#include <assert.h>

#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.h"
#endif

#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
#include "src/common/primitive_cache.hpp"
#endif

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "tests/test_thread.hpp"

#include "cpu/platform.hpp"

int check_pd_cache(dnnl_primitive_desc_t pd) {
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    int capacity = 0;
    DNN_SAFE(dnnl_get_primitive_cache_capacity(&capacity), CRIT);
    if (capacity && !dnnl::impl::is_pd_in_cache(pd)) {
        BENCHDNN_PRINT(0, "error: %s\n",
                "primitive descriptor is expected to be fetched from "
                "the primitive cache");
        return FAIL;
    }
#endif
    return OK;
}

int check_primitive_cache(dnnl_primitive_t p) {
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    int capacity = 0;
    DNN_SAFE(dnnl_get_primitive_cache_capacity(&capacity), CRIT);
    if (capacity && !dnnl::impl::is_primitive_in_cache(p)) {
        BENCHDNN_PRINT(0, "error: %s\n",
                "primitive is expected to be fetched from the primitive "
                "cache");
        return FAIL;
    }
#endif
    return OK;
}

float round_to_nearest_representable(dnnl_data_type_t dt, float value) {
    switch (dt) {
        case dnnl_f32: break;
        case dnnl_bf16: value = (float)dnnl::impl::bfloat16_t(value); break;
        case dnnl_f16: value = (float)dnnl::impl::float16_t(value); break;
        case dnnl_s32:
        case dnnl_s8:
        case dnnl_u8: value = maybe_saturate(dt, value); break;
        default: SAFE(FAIL, CRIT);
    }

    return value;
}

// Engine kind used to run oneDNN primitives for testing
dnnl_engine_kind_t engine_tgt_kind = dnnl_cpu;
// Engine index used to run oneDNN primitives for testing
size_t engine_index = 0;
// CPU ISA specific hints : none by default
isa_hints_t hints {isa_hints_t::none};

memory_kind_ext_t memory_kind {default_memory_kind};

void init_isa_settings() {
    if (hints.get() == isa_hints_t::no_hints)
        DNN_SAFE_V(dnnl_set_cpu_isa_hints(dnnl_cpu_isa_no_hints));
    else if (hints.get() == isa_hints_t::prefer_ymm)
        DNN_SAFE_V(dnnl_set_cpu_isa_hints(dnnl_cpu_isa_prefer_ymm));
    else {
        // Do nothing when hints == none
        assert(hints.get() == isa_hints_t::none);
    }
}

args_t &args_t::set(int arg, const dnn_mem_t &mem) {
    args_.emplace_back(arg, &mem);
    return *this;
}

args_t &args_t::set(
        const std::vector<int> &args, const std::vector<dnn_mem_t> &mems) {
    assert(args.size() == mems.size());
    for (size_t i = 0; i < mems.size(); ++i)
        args_.emplace_back(args[i], &mems[i]);
    return *this;
}

const dnn_mem_t &args_t::find(int arg) const {
    static dnn_mem_t empty_stub;
    for (const auto &e : args_) {
        if (e.first == arg) return *(e.second);
    }
    return empty_stub;
}

// Unmap before passing the memory to execute
void execute_unmap_args(
        const args_t &args, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    dnnl_args.resize(args.size());
    for (int i = 0; i < args.size(); ++i) {
        if (args.dnn_mem(i).is_mapped()) args.dnn_mem(i).unmap();

        dnnl_args[i].arg = args.arg(i);
        dnnl_args[i].memory = args.dnn_mem(i).m_;
    }
}

// Map the memory back after execute
void execute_map_args(const args_t &args) {
    for (int i = 0; i < args.size(); ++i)
        if (!args.dnn_mem(i).is_mapped()) args.dnn_mem(i).map();
}

int execute_and_wait(perf_function_t &exec_func, const dnnl_engine_t &engine,
        const args_t &args) {
    stream_t stream(engine);
    std::vector<dnnl_exec_arg_t> dnnl_args;
    execute_unmap_args(args, dnnl_args);

    DNN_SAFE(exec_func(stream, dnnl_args), CRIT);
    DNN_SAFE(dnnl_stream_wait(stream), CRIT);

    execute_map_args(args);

    if (is_bench_mode(CORR)) {
        for (int i = 0; i < args.size(); ++i) {
            SAFE(check_zero_padding(args.dnn_mem(i), args.arg(i)), WARN);
        }
    }

    return OK;
}

dnnl_status_t primitive_executor(dnnl_primitive_t prim,
        const dnnl_stream_t &stream,
        const std::vector<dnnl_exec_arg_t> &dnnl_args) {
    return dnnl_primitive_execute(
            prim, stream, (int)dnnl_args.size(), dnnl_args.data());
}

int execute_and_wait(dnnl_primitive_t prim, const args_t &args) {
    perf_function_t exec_func = std::bind(&primitive_executor, prim,
            std::placeholders::_1, std::placeholders::_2);

    const_dnnl_primitive_desc_t pd;
    dnnl_engine_t engine;

    DNN_SAFE(dnnl_primitive_get_primitive_desc(prim, &pd), CRIT);

    DNN_SAFE(
            dnnl_primitive_desc_query(pd, dnnl_query_engine, 0, &engine), CRIT);

    return execute_and_wait(exec_func, engine, args);
}

inline bool should_stop(const timer::timer_t &t) {
    const bool stop = false
            || (fix_times_per_prb && t.times() >= fix_times_per_prb)
            || (!fix_times_per_prb && t.total_ms() >= max_ms_per_prb
                    && t.times() >= min_times_per_prb);
    return stop;
}

dnnl_engine_kind_t get_engine_kind(const dnnl_engine_t &engine) {
    dnnl_engine_kind_t engine_kind = dnnl_any_engine;
    DNN_SAFE_V(dnnl_engine_get_kind(engine, &engine_kind));
    return engine_kind;
}

inline int measure_perf_individual(timer::timer_t &t, dnnl_stream_t stream,
        perf_function_t &perf_func, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    t.reset();
    while (true) {
        DNN_SAFE(perf_func(stream, dnnl_args), WARN);
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

inline int measure_perf_aggregate(timer::timer_t &t, dnnl_stream_t stream,
        perf_function_t &perf_func, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    const int max_batch_times = 10000;

    // Warm-up run, this is not measured due to possibility the associated
    // kernel has not been built and skews the results.
    DNN_SAFE(perf_func(stream, dnnl_args), WARN);
    DNN_SAFE(dnnl_stream_wait(stream), WARN);

    int cur_batch_times
            = fix_times_per_prb ? fix_times_per_prb : min_times_per_prb;

    t.reset();

    bool is_first_loop = true;
    while (true) {
        for (int i = 0; i < cur_batch_times; i++) {
            DNN_SAFE(perf_func(stream, dnnl_args), WARN);
        }
        DNN_SAFE(dnnl_stream_wait(stream), WARN);
        t.stamp(cur_batch_times);

        if (should_stop(t)) break;

        // Adjust cur_batch_times after the first batch run
        if (is_first_loop) {
            double ms_min = t.ms(timer::timer_t::min);
            // Heuristic: try to use ~5 batch runs for the whole benchmark
            int batch_times_heuristic = (ms_min == 0.0)
                    ? INT_MAX
                    : MAX2(1,
                            (int)((max_ms_per_prb - t.total_ms()) / ms_min
                                    / 5));
            cur_batch_times = MIN2(max_batch_times, batch_times_heuristic);
            is_first_loop = false;
        }
    }
    return OK;
}

int measure_perf(timer::timer_t &t, perf_function_t &perf_func, args_t &args) {
    int ret = OK;
    if (is_bench_mode(PERF)) {
        const auto &engine = get_test_engine();
        stream_t stream(engine);
        std::vector<dnnl_exec_arg_t> dnnl_args;
        execute_unmap_args(args, dnnl_args);

        // For non-DPCPP CPU: measure individual iterations.
        // For DPCPP CPU and GPU: measure iterations in batches to hide driver
        // overhead. DPCPP CPU follows the model of GPU, thus, handled similar.
        if (is_cpu() && !is_sycl_engine(engine))
            ret = measure_perf_individual(t, stream, perf_func, dnnl_args);
        else
            ret = measure_perf_aggregate(t, stream, perf_func, dnnl_args);

        if (ret == OK) execute_map_args(args);
    }
    return ret;
}

int measure_perf(timer::timer_t &t, dnnl_primitive_t prim, args_t &args) {
    perf_function_t perf_func = std::bind(&primitive_executor, prim,
            std::placeholders::_1, std::placeholders::_2);

    return measure_perf(t, perf_func, args);
}

void maybe_prepare_runtime_scales(dnn_mem_t &scales_m,
        const attr_t::scale_t &scale, int64_t scale_cnt, const float *scales) {
    if (!scale.runtime) return;

    const int64_t count = scale.policy == policy_t::COMMON ? 1 : scale_cnt;

    scales_m = dnn_mem_t(1, &count, dnnl_f32, tag::x, get_test_engine());
    for (int64_t c = 0; c < count; ++c)
        ((float *)scales_m)[c] = scales[c];
}

void maybe_prepare_runtime_zero_points(dnn_mem_t &zero_points_m,
        const attr_t &attr, int arg, int64_t count,
        const int32_t *zero_points) {
    if (!attr.zero_points.runtime(arg)) return;

    const auto e = attr.zero_points.get(arg);
    const int64_t cnt = e.policy == policy_t::COMMON ? 1 : count;

    zero_points_m = dnn_mem_t(1, &cnt, dnnl_s32, tag::x, get_test_engine());
    for (int64_t c = 0; c < cnt; ++c)
        ((int32_t *)zero_points_m)[c] = zero_points[c];
}

std::vector<float> prepare_po_vals(const dnn_mem_t &dst_m, const args_t &args,
        const std::vector<std::pair<int, int>> &v_po_masks,
        const size_t dst_off) {
    std::vector<float> v_vals(v_po_masks.size());

    for (size_t d = 0; d < v_po_masks.size(); ++d) {
        const auto po_offset
                = dst_m.get_scale_idx(dst_off, v_po_masks[d].second);
        const float val = args.find(v_po_masks[d].first).get_elem(po_offset);
        v_vals[d] = val;
    }
    return v_vals;
}

bool check_md_consistency_with_tag(
        const dnnl_memory_desc_t &md, const std::string &tag) {
    dnnl_memory_desc_t md_new_tag;
    SAFE(init_md(&md_new_tag, md.ndims, md.dims, md.data_type, tag), CRIT);
    return dnnl_memory_desc_equal(&md_new_tag, &md);
}

void check_known_skipped_case_common(
        const std::vector<dnnl_data_type_t> &v_dt, dir_t dir, res_t *res) {
    if (benchdnn_stat.tests < test_start) {
        res->state = SKIPPED;
        res->reason = SKIP_START;
        return;
    }

    bool has_bf16_support = is_gpu();
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    using namespace dnnl::impl::cpu::platform;
    has_bf16_support = has_bf16_support
            || (is_cpu() && has_data_type_support(dnnl_bf16));
#endif

    for (const auto &i_dt : v_dt) {
        // bf16 is supported on AVX512-CORE+
        if (!has_bf16_support && i_dt == dnnl_bf16) {
            res->state = SKIPPED, res->reason = DATA_TYPE_NOT_SUPPORTED;
            break;
        }
        // f16 is supported on GPU only
        if (i_dt == dnnl_f16 && is_cpu()) {
            res->state = SKIPPED, res->reason = DATA_TYPE_NOT_SUPPORTED;
            break;
        }
        // f16 is supported for inference only
        if (i_dt == dnnl_f16 && (dir & FLAG_BWD)) {
            res->state = SKIPPED, res->reason = DATA_TYPE_NOT_SUPPORTED;
            break;
        }
        // cuda supports only f32, f16 and s8 data types
        if (is_nvidia_gpu()
                && (i_dt == dnnl_bf16 || i_dt == dnnl_u8 || i_dt == dnnl_s32)) {
            res->state = SKIPPED, res->reason = DATA_TYPE_NOT_SUPPORTED;
            break;
        }
    }
}

// Binary MAX, MIN and comparison operations post-ops may return different
// results for different backends when NaN is one of inputs. Depending on its
// position and implementation, either first or second operand may be returned.
// There isn't a single standard, thus, marking such cases as SKIPPED with
// KNOWN_LIMITATION reason.
void check_binary_post_ops(const attr_t &attr, res_t *res) {
    if (attr.is_def()) return;

    using pk_t = attr_t::post_ops_t::kind_t;
    const auto &po = attr.post_ops;
    if (!po.is_def()) {
        for (int idx = 0; idx < po.len(); ++idx) {
            const auto &e = po.entry[idx];
            if (!e.is_binary_kind()) continue;
            if (e.kind == pk_t::MAX || e.kind == pk_t::MIN || e.kind == pk_t::GE
                    || e.kind == pk_t::GT || e.kind == pk_t::LE
                    || e.kind == pk_t::LT || e.kind == pk_t::EQ
                    || e.kind == pk_t::NE) {
                res->state = SKIPPED, res->reason = KNOWN_LIMITATION;
                break;
            }
        }
    }
}

void check_sum_post_ops(
        const attr_t &attr, res_t *res, dnnl_data_type_t dst_dt) {
    const auto &po = attr.post_ops;
    if (!po.is_def()) {
        const int first_sum_idx = po.find(attr_t::post_ops_t::SUM);
        if (first_sum_idx == -1) return;
        const auto sum_dt = po.entry[first_sum_idx].sum.dt;

        for (int idx = 0; idx < po.len(); ++idx) {
            const auto &e = po.entry[idx];
            if (e.is_sum_kind()) {
                // Sum with zero-point is not supported on GPU
                if (is_gpu() && e.sum.zero_point != 0) {
                    res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                    break;
                }
                // Each sum must have same data on CPU
                if (is_cpu() && e.sum.dt != sum_dt) {
                    res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                    break;
                }
                // Sum must have data type with the same size like dst on both
                if (dst_dt != dnnl_data_type_undef
                        && sum_dt != dnnl_data_type_undef
                        && sizeof_dt(dst_dt) != sizeof_dt(e.sum.dt)) {
                    res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                    return;
                }
            }
        }
    }
}

// Check ensures that attributes don't cause implementation fallback
int check_same_pd(res_t *res, const dnnl_primitive_desc_t &pd_no_attr) {
    const std::string pd_no_attr_name = query_impl_info(pd_no_attr);
    if (res->impl_name != pd_no_attr_name) {
        res->state = FAILED;
        BENCHDNN_PRINT(0,
                "ERROR: attributes usage caused implementation fallback from "
                "[%s] to [%s] \n",
                pd_no_attr_name.c_str(), res->impl_name.c_str());
        return FAIL;
    }
    return OK;
}

bool is_cpu(const dnnl_engine_t &engine) {
    return get_engine_kind(engine) == dnnl_cpu;
}

bool is_gpu(const dnnl_engine_t &engine) {
    return get_engine_kind(engine) == dnnl_gpu;
}

bool is_sycl_engine(const dnnl_engine_t &engine) {
    if (is_cpu(engine)) return DNNL_CPU_RUNTIME == DNNL_RUNTIME_DPCPP;
    if (is_gpu(engine)) return DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP;
    return false;
}

bool is_opencl_engine(const dnnl_engine_t &engine) {
    if (is_gpu(engine)) return DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL;
    return false;
}

bool is_nvidia_gpu(const dnnl_engine_t &engine) {
#ifdef DNNL_WITH_SYCL
    if (!is_gpu(engine)) return false;
    constexpr int nvidia_vendor_id = 0x10DE;
    auto eng = dnnl::engine(engine, true);
    auto device = dnnl::sycl_interop::get_device(eng);
    const auto eng_vendor_id
            = device.get_info<cl::sycl::info::device::vendor_id>();
    return eng_vendor_id == nvidia_vendor_id;
#endif
    return false;
}

bool is_nvidia_eltwise_ok(
        dir_t dir, attr_t::post_ops_t::kind_t alg, float alpha) {
    using pk_t = attr_t::post_ops_t::kind_t;
    switch (alg) {
        case pk_t::BRELU: return true;
        case pk_t::ELU: return (dir & FLAG_FWD);
        case pk_t::LOGISTIC: return (dir & FLAG_FWD);
        case pk_t::TANH: return (dir & FLAG_FWD);
        case pk_t::RELU: return alpha == 0.f;
        // TODO: can be easily supported by Nvidia backend
        // case pk_t::ELU_DST: return true;
        // case pk_t::LOGISTIC_DST: return true;
        // case pk_t::TANH_DST: return true;
        // case pk_t::RELU_DST: return alpha == 0.f;
        default: return false;
    };
}

int init_md(dnnl_memory_desc_t *md, int ndims, const dnnl_dims_t dims,
        dnnl_data_type_t data_type, const std::string &tag_,
        const dims_t &strides_) {
    const bool use_strides = !strides_.empty();
    // Ignore tag_ in case strides_ are explicitly provided
    if (use_strides) {
        std::vector<dnnl_dim_t> strides(strides_);
        DNN_SAFE(dnnl_memory_desc_init_by_strides(
                         md, ndims, dims, data_type, strides.data()),
                CRIT);
        return OK;
    }
    auto tag = normalize_tag(tag_, ndims);
    if (tag == tag::undef || tag == tag::any || ndims == 0) {
        dnnl_format_tag_t enum_tag = (tag == tag::undef || ndims == 0)
                ? dnnl_format_tag_undef
                : dnnl_format_tag_any;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(
                         md, ndims, dims, data_type, enum_tag),
                CRIT);
        return OK;
    }

    // Copy to temporary to handle dims == md->dims case.
    dnnl_dims_t tmp_dims;
    std::copy(dims, dims + ndims, tmp_dims);

    *md = dnnl_memory_desc_t();
    md->ndims = ndims;
    std::copy(tmp_dims, tmp_dims + ndims, md->dims);
    md->data_type = data_type;
    md->format_kind = dnnl_blocked;

    // Parse dimensions and their block sizes starting from the innermost one.
    std::vector<std::pair<int, int>> dim_blocks;
    int pos = (int)tag.size() - 1;
    int ndims_from_tag = -1;
    while (pos >= 0) {
        int pos0 = pos;

        --pos;
        while (pos >= 0 && std::isdigit(tag[pos]))
            pos--;

        int dim_idx = std::tolower(tag[pos0]) - 'a';
        if (dim_idx >= ndims) return FAIL;
        ndims_from_tag = MAX2(dim_idx + 1, ndims_from_tag);
        int block_str_len = pos0 - pos - 1;
        int block = (block_str_len == 0)
                ? 1
                : std::stoi(tag.substr(pos + 1, block_str_len));
        dim_blocks.emplace_back(dim_idx, block);
    }
    if (ndims_from_tag != ndims) return FAIL;

    auto &blk = md->format_desc.blocking;

    // Compute strides and fill inner block sizes/indices.
    dnnl_dim_t stride = 1;
    dnnl_dims_t full_inner_blks;
    std::fill(full_inner_blks, full_inner_blks + ndims, 1);
    for (auto &p : dim_blocks) {
        int dim_idx = p.first;
        int block = p.second;
        if (block == 1) {
            assert(blk.strides[dim_idx] == 0);
            blk.strides[dim_idx] = stride;

            dnnl_dim_t fib = full_inner_blks[dim_idx];
            dnnl_dim_t padded_dim = md->dims[dim_idx] == DNNL_RUNTIME_DIM_VAL
                    ? DNNL_RUNTIME_DIM_VAL
                    : (md->dims[dim_idx] + fib - 1) / fib * fib;
            md->padded_dims[dim_idx] = padded_dim;
            if (padded_dim == DNNL_RUNTIME_DIM_VAL)
                stride = DNNL_RUNTIME_DIM_VAL;
            else
                stride *= (padded_dim / fib);
        } else {
            full_inner_blks[dim_idx] *= block;
            blk.inner_blks[blk.inner_nblks] = block;
            blk.inner_idxs[blk.inner_nblks] = dim_idx;
            blk.inner_nblks++;
            stride *= block;
        }
    }

    // Inner block sizes/indices are stored from the outermost to the innermost
    // so need to reverse them.
    std::reverse(blk.inner_blks, blk.inner_blks + blk.inner_nblks);
    std::reverse(blk.inner_idxs, blk.inner_idxs + blk.inner_nblks);

    return OK;
}

#if defined(_WIN32) && !defined(__GNUC__)
#include "windows.h"

static size_t get_cpu_ram_size() {
    MEMORYSTATUSEX s {};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    return s.ullTotalPhys;
}
#elif defined(__APPLE__) || defined(__FreeBSD__)
#include <unistd.h>
#include <sys/sysctl.h>

static size_t get_cpu_ram_size() {
#ifdef __APPLE__
    int query_ram[] = {CTL_HW, HW_MEMSIZE};
#else
    int query_ram[] = {CTL_HW, HW_PHYSMEM};
#endif
    int query_ram_len = sizeof(query_ram) / sizeof(*query_ram);
    size_t totalram = 0;
    size_t length = sizeof(totalram);

    sysctl(query_ram, query_ram_len, &totalram, &length, NULL, 0);
    return totalram;
}
#else
#include <sys/sysinfo.h>

static size_t get_cpu_ram_size() {
    struct sysinfo s {};
    sysinfo(&s);
    return s.totalram;
}
#endif

static size_t get_gpu_ram_size() {
    // XXX: create a tmp engine to query what we need.
    // It will be removed in the future as part of switching back
    // to the global engine.
    engine_t eng_tmp(engine_tgt_kind);
    dnnl::engine eng(eng_tmp, true);
    if (eng.get_kind() != dnnl::engine::kind::gpu) return 0;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_int status = CL_SUCCESS;
    // Get single device attached to the engine.
    engine_t engine_tgt(engine_tgt_kind);
    cl_device_id ocl_device = dnnl::ocl_interop::get_device(eng);

    cl_ulong ram_size = 0;
    status = clGetDeviceInfo(ocl_device, CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(cl_ulong), &ram_size, nullptr);
    if (status == CL_SUCCESS) return (size_t)ram_size;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto sycl_dev = dnnl::sycl_interop::get_device(eng);
    return (size_t)sycl_dev.get_info<cl::sycl::info::device::global_mem_size>();
#endif
    return 0;
}

static int validate_mem_size(size_t total_mem_size) {
    static uint64_t cpu_device_capacity = get_cpu_ram_size();
    static uint64_t gpu_device_capacity = get_gpu_ram_size();

    const uint64_t devices_max_capacity = is_cpu()
            ? cpu_device_capacity
            : MIN2(cpu_device_capacity, gpu_device_capacity);

    // 0.75f is taken randomly and is subject to change in future.
    const double capacity_factor = 0.75;
    const double benchdnn_limit = capacity_factor * devices_max_capacity;
    assert(benchdnn_limit > 0);

    const bool fits_device_ram = total_mem_size <= benchdnn_limit;
    auto GB = [](double bytes) { return bytes / powf(2, 30); };

    if (!fits_device_ram)
        BENCHDNN_PRINT(2, "%s\n", "benchdnn: not enough RAM for a problem.");

    BENCHDNN_PRINT((!fits_device_ram ? 2 : 6),
            "Requested: %g GB, benchdnn limit: %g GB, CPU RAM capacity: %g GB, "
            "GPU RAM capacity: %g GB\n",
            GB(total_mem_size), GB(benchdnn_limit), GB(cpu_device_capacity),
            GB(gpu_device_capacity));

    return fits_device_ram ? OK : FAIL;
}

static size_t get_md_size(
        const dnnl_memory_desc_t *md, bool add_ref_size = false) {
    const auto mem_size = dnnl_memory_desc_get_size(md);
    // runtime mem size is not defined
    if (mem_size == 0 || mem_size == DNNL_RUNTIME_SIZE_VAL) return 0;
    if (!add_ref_size) return mem_size;

    // reference memories are always fp32, hence need rescaling factor
    size_t ref_mem_factor = 1;
    if (md->data_type != dnnl_data_type_undef)
        ref_mem_factor = ::sizeof_dt(dnnl_f32) / ::sizeof_dt(md->data_type);
    // all memory is mapped once it is created and unmapped only before
    // primitive execution. Device memory requires additional buffer for mapped
    // memory.
    // XXX: In DPC++ build oneDNN uses USM memory, which shouldn't require an
    // additional buffer, so mapped_mem_factor should be equal to 0 for DPC++.
    // However due to a driver issue oneDNN pretends that shared USM is not
    // accessible on the host, hence map will allocate an extra memory.
    const size_t mapped_mem_factor = engine_tgt_kind == dnnl_cpu ? 0 : 1;
    return (1 + mapped_mem_factor + ref_mem_factor) * mem_size;
}

static size_t get_memory_bytes(const_dnnl_primitive_desc_t const_pd,
        bool want_input, bool add_ref_size = false) {
    const int n_idx = dnnl_primitive_desc_query_s32(const_pd,
            want_input ? dnnl_query_num_of_inputs_s32
                       : dnnl_query_num_of_outputs_s32,
            0);

    dnnl_prop_kind_t prop_kind = dnnl_prop_kind_undef;
    dnnl_primitive_desc_query(const_pd, dnnl_query_prop_kind, 0, &prop_kind);
    const bool is_fwd = prop_kind == dnnl_forward_training
            || prop_kind == dnnl_forward_inference
            || prop_kind == dnnl_prop_kind_undef;

#define MD(name) dnnl_query_##name##_md
    std::vector<dnnl_query_t> query_fwd_in_mds {MD(src), MD(weights)};
    std::vector<dnnl_query_t> query_fwd_out_mds {MD(dst), MD(workspace)};
    std::vector<dnnl_query_t> query_bwd_in_mds {
            MD(src), MD(weights), MD(dst), MD(diff_dst), MD(workspace)};
    std::vector<dnnl_query_t> query_bwd_out_mds {
            MD(diff_src), MD(diff_weights)};
    std::vector<dnnl_query_t> query_mds = is_fwd
            ? (want_input ? query_fwd_in_mds : query_fwd_out_mds)
            : (want_input ? query_bwd_in_mds : query_bwd_out_mds);
#undef MD

    size_t total_mem_size = 0;
    for_(const auto query : query_mds)
    for (int idx = 0; idx < n_idx; ++idx) {
        const auto md = dnnl_primitive_desc_query_md(const_pd, query, idx);
        total_mem_size += get_md_size(md, add_ref_size);
    }
    return total_mem_size;
}

int check_mem_size(const dnnl_memory_desc_t &md) {
    if (!mem_check) return OK;

    size_t total_mem_size = dnnl_memory_desc_get_size(&md);

    return validate_mem_size(total_mem_size);
}

int check_mem_size(const_dnnl_primitive_desc_t const_pd) {
    if (!mem_check) return OK;

    bool add_ref_size = true;
    bool inputs = true;
    bool outputs = !inputs;
    size_t total_mem_size = get_memory_bytes(const_pd, inputs, add_ref_size)
            + get_memory_bytes(const_pd, outputs, add_ref_size);

    const auto scratchpad = dnnl_primitive_desc_query_md(
            const_pd, dnnl_query_scratchpad_md, 0);
    total_mem_size += get_md_size(scratchpad, add_ref_size);

    int64_t library_internal_mem_size = 0;
    dnnl_primitive_desc_query(const_pd, dnnl_query_memory_consumption_s64, 0,
            &library_internal_mem_size);
    total_mem_size += library_internal_mem_size;

    return validate_mem_size(total_mem_size);
}

int get_memory_footprint(const_dnnl_primitive_desc_t const_pd, res_t *res) {
    res->ibytes = get_memory_bytes(const_pd, /* want_input = */ true);
    res->obytes = get_memory_bytes(const_pd, /* want_input = */ false);

    // Update read bytes with dst bytes in case of sum post-op.
    const_dnnl_primitive_attr_t const_attr;
    DNN_SAFE(dnnl_primitive_desc_get_attr(const_pd, &const_attr), WARN);

    const_dnnl_post_ops_t const_attr_po;
    DNN_SAFE(
            dnnl_primitive_attr_get_post_ops(const_attr, &const_attr_po), WARN);

    auto po_len = dnnl_post_ops_len(const_attr_po);
    for (int idx = 0; idx < po_len; ++idx) {
        const auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
        if (kind == dnnl_sum) {
            const auto dst_md = dnnl_primitive_desc_query_md(
                    const_pd, dnnl_query_dst_md, 0);
            res->ibytes += get_md_size(dst_md);
        }
    }
    return OK;
}

memory_kind_ext_t str2memory_kind(const char *str) {
#define CASE(param) \
    if (!strcasecmp(#param, str)) return memory_kind_ext_t::param

    CASE(usm);
    CASE(buffer);
    CASE(usm_device);
    CASE(usm_shared);

#undef CASE

    assert(!"not expected");
    return memory_kind_ext_t::usm;
}

engine_t::engine_t(dnnl_engine_kind_t engine_kind) : is_owner_(true) {
    size_t idx = engine_kind == dnnl_cpu ? 0 : engine_index;
    DNN_SAFE_V(dnnl_engine_create(&engine_, engine_kind, idx));
}

engine_t::engine_t(dnnl_engine_t engine) : engine_(engine), is_owner_(false) {}

engine_t::engine_t(const engine_t &other) {
    is_owner_ = other.is_owner_;

    if (!is_owner_) {
        engine_ = other.engine_;
        return;
    }

    dnnl_engine_kind_t engine_kind;
    DNN_SAFE_V(dnnl_engine_get_kind(other.engine_, &engine_kind));

    if (engine_kind == dnnl_cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        void *dev;
        void *ctx;
        DNN_SAFE_V(dnnl_sycl_interop_engine_get_device(other.engine_, &dev));
        DNN_SAFE_V(dnnl_sycl_interop_engine_get_context(other.engine_, &ctx));
        DNN_SAFE_V(dnnl_sycl_interop_engine_create(&engine_, dev, ctx));
#else
        DNN_SAFE_V(dnnl_engine_create(&engine_, dnnl_cpu, 0));
#endif
    } else if (engine_kind == dnnl_gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        cl_device_id dev;
        cl_context ctx;
        DNN_SAFE_V(dnnl_ocl_interop_get_device(other.engine_, &dev));
        DNN_SAFE_V(dnnl_ocl_interop_engine_get_context(other.engine_, &ctx));
        DNN_SAFE_V(dnnl_ocl_interop_engine_create(&engine_, dev, ctx));
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        void *dev;
        void *ctx;
        DNN_SAFE_V(dnnl_sycl_interop_engine_get_device(other.engine_, &dev));
        DNN_SAFE_V(dnnl_sycl_interop_engine_get_context(other.engine_, &ctx));
        DNN_SAFE_V(dnnl_sycl_interop_engine_create(&engine_, dev, ctx));
#endif
    } else {
        assert(!"unsupported engine kind");
    }
}

engine_t::~engine_t() {
    if (is_owner_) DNN_SAFE_V(dnnl_engine_destroy(engine_));
}

stream_t::stream_t(dnnl_engine_t engine) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (is_cpu(engine)) {
        SAFE_V(dnnl_threadpool_interop_stream_create(
                &stream_, engine, dnnl::testing::get_threadpool()));
        return;
    }
#endif
    DNN_SAFE_V(dnnl_stream_create(&stream_, engine, dnnl_stream_default_flags));
}

stream_t::~stream_t() {
    DNN_SAFE_V(dnnl_stream_destroy(stream_));
}

float reorder_rescale_factor() {
    float factor = 1.f;
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    factor = dnnl::impl::cpu::platform::s8s8_weights_scale_factor();
#endif
    return factor;
}
