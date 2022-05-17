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

#ifndef DNNL_COMMON_HPP
#define DNNL_COMMON_HPP

#include <functional>
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "oneapi/dnnl/dnnl.h"
#include "src/common/bfloat16.hpp"
#include "src/common/float16.hpp"
#include "src/common/nstl.hpp"

int check_pd_cache(dnnl_primitive_desc_t pd);
int check_primitive_cache(dnnl_primitive_t p);

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_debug.hpp"

#define for_ for

#define DNN_SAFE(f, s) \
    do { \
        dnnl_status_t status__ = f; \
        if (status__ != dnnl_success) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, \
                        status2str(status__), (int)status__); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            return FAIL; \
        } \
    } while (0)

#define DNN_SAFE_V(f) \
    do { \
        dnnl_status_t status__ = f; \
        if (status__ != dnnl_success) { \
            BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                    __PRETTY_FUNCTION__, __LINE__, STRINGIFY(f), \
                    status2str(status__), (int)status__); \
            fflush(0); \
            exit(2); \
        } \
    } while (0)

/* aux */
using bfloat16_t = dnnl::impl::bfloat16_t;
using float16_t = dnnl::impl::float16_t;
template <dnnl_data_type_t>
struct prec_traits;
template <>
struct prec_traits<dnnl_bf16> {
    typedef bfloat16_t type;
};
template <>
struct prec_traits<dnnl_f16> {
    typedef float16_t type;
};
template <>
struct prec_traits<dnnl_f32> {
    typedef float type;
};
template <>
struct prec_traits<dnnl_s32> {
    typedef int32_t type;
};
template <>
struct prec_traits<dnnl_s8> {
    typedef int8_t type;
};
template <>
struct prec_traits<dnnl_u8> {
    typedef uint8_t type;
};

#define CASE_ALL(dt) \
    switch (dt) { \
        CASE(dnnl_bf16); \
        CASE(dnnl_f16); \
        CASE(dnnl_f32); \
        CASE(dnnl_s32); \
        CASE(dnnl_s8); \
        CASE(dnnl_u8); \
        default: assert(!"bad data_type"); \
    }

inline size_t sizeof_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: return sizeof(typename prec_traits<dt>::type);

    CASE_ALL(dt);

#undef CASE
    return 0;
}

/* std::numeric_limits::digits functionality */
inline int digits_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::digits;

    CASE_ALL(dt);

#undef CASE
    return 0;
}

inline float epsilon_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::epsilon();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

inline float lowest_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::lowest();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

inline float max_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::max();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

#undef CASE_ALL

#define BENCHDNN_S32_TO_F32_SAT_CONST 2147483520.f

template <dnnl_data_type_t dt>
inline float saturate_and_round(float val) {
    const float dt_max = max_dt(dt);
    const float dt_min = (float)dnnl::impl::nstl::numeric_limits<
            typename prec_traits<dt>::type>::lowest();
    if (dt == dnnl_s32 && val >= max_dt(dnnl_s32)) return max_dt(dnnl_s32);
    if (val > dt_max) val = dt_max;
    if (val < dt_min) val = dt_min;
    return mxcsr_cvt(val);
}

inline bool is_integral_dt(dnnl_data_type_t dt) {
    return dt == dnnl_s32 || dt == dnnl_s8 || dt == dnnl_u8;
}

inline float maybe_saturate(dnnl_data_type_t dt, float value) {
    if (!is_integral_dt(dt)) return value;

    switch (dt) {
#define CASE(dt) \
    case dt: return saturate_and_round<dt>(value);
        CASE(dnnl_s32);
        CASE(dnnl_s8);
        CASE(dnnl_u8);
#undef CASE
        default: assert(!"bad data_type");
    }
    return 0;
}

float round_to_nearest_representable(dnnl_data_type_t dt, float value);

extern dnnl_engine_kind_t engine_tgt_kind;
extern size_t engine_index;
extern isa_hints_t hints;

// Extended version of dnnl_sycl_interop_memory_kind_t enumeration.
enum class memory_kind_ext_t {
    usm, // Same as dnnl_sycl_interop_usm
    buffer, // Same as dnnl_sycl_interop_buffer
    usm_device, // USM allocated via malloc_device()
    usm_shared, // USM allocated via malloc_shared()
};

const memory_kind_ext_t default_memory_kind = memory_kind_ext_t::usm;

extern memory_kind_ext_t memory_kind;

void init_isa_settings();

inline const char *query_impl_info(const_dnnl_primitive_desc_t pd) {
    const char *str;
    dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &str);
    return str;
}

struct dnn_mem_t;

struct args_t {
    args_t &set(int arg, const dnn_mem_t &mem);
    args_t &set(
            const std::vector<int> &args, const std::vector<dnn_mem_t> &mems);
    void clear() { args_.clear(); }

    int size() const { return (int)args_.size(); }

    const dnn_mem_t &find(int arg) const;

    int arg(int index) const { return args_[index].first; }
    const dnn_mem_t &dnn_mem(int index) const { return *args_[index].second; }

private:
    std::vector<std::pair<int, const dnn_mem_t *>> args_;
};

template <typename T>
struct dnnl_api_traits;
//{
//    static void destroy(T t) {}
//};

template <>
struct dnnl_api_traits<dnnl_primitive_t> {
    static void destroy(dnnl_primitive_t t) {
        DNN_SAFE_V(dnnl_primitive_destroy(t));
    }
};

template <>
struct dnnl_api_traits<dnnl_primitive_desc_t> {
    static void destroy(dnnl_primitive_desc_t t) {
        DNN_SAFE_V(dnnl_primitive_desc_destroy(t));
    }
};

template <>
struct dnnl_api_traits<dnnl_primitive_attr_t> {
    static void destroy(dnnl_primitive_attr_t t) {
        DNN_SAFE_V(dnnl_primitive_attr_destroy(t));
    }
};

// Generic class providing RAII support for DNNL objects in benchdnn
template <typename T>
struct benchdnn_dnnl_wrapper_t {
    benchdnn_dnnl_wrapper_t(T t = nullptr) : t_(t) {
        static_assert(std::is_pointer<T>::value, "T is not a pointer type.");
    }

    benchdnn_dnnl_wrapper_t(benchdnn_dnnl_wrapper_t &&rhs) {
        T t = rhs.release();
        t_ = t;
    }

    ~benchdnn_dnnl_wrapper_t() { do_destroy(); }

    T release() {
        T tmp = t_;
        t_ = nullptr;
        return tmp;
    }

    void reset(T t) {
        do_destroy();
        t_ = t;
    }

    operator T() const { return t_; }

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(benchdnn_dnnl_wrapper_t);

private:
    T t_;

    void do_destroy() {
        if (t_) { dnnl_api_traits<T>::destroy(t_); }
    }
};

// Constructs a wrapper object (providing RAII support)
template <typename T>
benchdnn_dnnl_wrapper_t<T> make_benchdnn_dnnl_wrapper(T t) {
    return benchdnn_dnnl_wrapper_t<T>(t);
}

struct engine_t {
    engine_t(dnnl_engine_kind_t engine_kind);
    engine_t(dnnl_engine_t engine);
    engine_t(const engine_t &other);
    ~engine_t();
    operator dnnl_engine_t() const { return engine_; }

private:
    engine_t &operator=(engine_t &other) = delete;
    dnnl_engine_t engine_;
    bool is_owner_;
};

struct stream_t {
    stream_t(dnnl_engine_t engine);
    ~stream_t();
    operator dnnl_stream_t() const { return stream_; }

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(stream_t);
    dnnl_stream_t stream_;
};

// Engine used to run oneDNN primitives for testing.
inline const engine_t &get_test_engine() {
    static const engine_t instance(engine_tgt_kind);
    return instance;
}

// Engine used to run reference implementations (fast-ref-gpu option).
inline const engine_t &get_cpu_engine() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    fprintf(stderr,
            "CPU engine is not available for GPU only configurations\n");
    SAFE_V(FAIL);
    assert(!"unexpected");
#endif
    static const engine_t instance(dnnl_cpu);
    return instance;
}

int get_memory_footprint(const_dnnl_primitive_desc_t pd, res_t *res);
int check_same_pd(res_t *res, const dnnl_primitive_desc_t &pd_no_attr);

template <typename op_desc_t>
int check_pd_w_and_wo_attr(
        res_t *res, const attr_t &attr, const op_desc_t &op_desc) {
    if (attr_same_pd_check && !attr.is_def()) {
        dnnl_primitive_desc_t pd_no_attr {};
        dnnl_primitive_attr_t dnnl_empty_attrs {};
        DNN_SAFE(dnnl_primitive_desc_create(&pd_no_attr, &op_desc,
                         dnnl_empty_attrs, get_test_engine(), nullptr),
                WARN);
        auto pd_no_attr_wrapper = make_benchdnn_dnnl_wrapper(pd_no_attr);
        SAFE(check_same_pd(res, pd_no_attr_wrapper), WARN);
    }
    return OK;
}

template <typename func_t, typename prb_t>
int init_prim(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &user_prim,
        const func_t &init_pd_func, prb_t *prb, res_t *res,
        dir_t dir = FLAG_FWD, const_dnnl_primitive_desc_t hint = nullptr) {
    dnnl_primitive_desc_t pd_ {};
    dnnl_primitive_t prim_ {};
    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pd;
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;

#ifndef DNNL_DISABLE_PRIMITIVE_CACHE

    // The first primitive creation using a temporary engine.
#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    // The idea is to create the requested primitive twice using different
    // engines but the same device and context in the case of OpenCL and DPCPP.
    // Rationale: make sure that the primitive cache is robust in the case
    // where CPU and GPU engines are re-created because this is a commonly
    // used scenario in the frameworks.
    engine_t engine(get_test_engine());
#else
    // The idea is to create the requested primitive twice using
    // different engines.
    // Rationale:
    // 1. Make sure that the primitive cache is robust for the cases when:
    //   - CPU engine is re-created
    //   - GPU engine is re-created for the same device but different context
    // These 2 cases are commonly used or expected to be used in the frameworks.
    // 2. (for GPU only) Identify context dependent parts in primitive
    // implementations, e.g. if a primitive implementation contains
    // a memory_storage_t (for scales, zero points or buffers), which depends
    // on a particular engine then it should fail at execution time.
    engine_t engine(engine_tgt_kind);
#endif

    SAFE(init_pd_func(engine, prb, pd_, res, dir, hint), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    DNN_SAFE(dnnl_primitive_create(&prim_, pd_), WARN);

    pd.reset(pd_);
    prim.reset(prim_);
#endif
    // The second (if the cache is enabled) primitive creation using
    // the global test engine.
    SAFE(init_pd_func(get_test_engine(), prb, pd_, res, dir, hint), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    // This primitive is expected to come from the cache.
    DNN_SAFE(dnnl_primitive_create(&prim_, pd_), WARN);

    pd.reset(pd_);
    prim.reset(prim_);

    SAFE(check_pd_cache(pd), WARN);
    SAFE(check_primitive_cache(prim), WARN);
    // Collect memory footprint for a given primitive descriptor.
    SAFE(get_memory_footprint(pd, res), WARN);

    user_prim.reset(prim.release());

    return OK;
}

typedef std::function<dnnl_status_t(
        const dnnl_stream_t &, const std::vector<dnnl_exec_arg_t> &)>
        perf_function_t;

int execute_and_wait(perf_function_t &exec_func, const dnnl_engine_t &engine,
        const args_t &args);
int execute_and_wait(dnnl_primitive_t prim, const args_t &args);

int measure_perf(timer::timer_t &t, perf_function_t &perf_func, args_t &args);
int measure_perf(timer::timer_t &t, dnnl_primitive_t prim, args_t &args);

void maybe_prepare_runtime_scales(dnn_mem_t &scales_m,
        const attr_t::scale_t &scale, int64_t scale_cnt, const float *scales);

void maybe_prepare_runtime_zero_points(dnn_mem_t &zero_points_m,
        const attr_t &attr, int arg, int64_t count, const int32_t *zero_points);

std::vector<float> prepare_po_vals(const dnn_mem_t &dst_m, const args_t &args,
        const std::vector<std::pair<int, int>> &v_po_masks,
        const size_t dst_off);

bool check_md_consistency_with_tag(
        const dnnl_memory_desc_t &md, const std::string &tag);

void check_known_skipped_case_common(
        const std::vector<dnnl_data_type_t> &v_dt, dir_t dir, res_t *res);
void check_binary_post_ops(const attr_t &attr, res_t *res);
void check_sum_post_ops(const attr_t &attr, res_t *res,
        dnnl_data_type_t dst_dt = dnnl_data_type_undef);

bool is_cpu(const dnnl_engine_t &engine = get_test_engine());
bool is_gpu(const dnnl_engine_t &engine = get_test_engine());
bool is_sycl_engine(const dnnl_engine_t &engine = get_test_engine());
bool is_opencl_engine(const dnnl_engine_t &engine = get_test_engine());
bool is_nvidia_gpu(const dnnl_engine_t &engine = get_test_engine());
bool is_nvidia_eltwise_ok(
        dir_t dir, attr_t::post_ops_t::kind_t alg, float alpha);
inline bool is_nvidia_eltwise_ok(
        dir_t dir, const attr_t::post_ops_t::entry_t &e) {
    return is_nvidia_eltwise_ok(dir, e.kind, e.eltwise.alpha);
}

int init_md(dnnl_memory_desc_t *md, int ndims, const dnnl_dims_t dims,
        dnnl_data_type_t data_type, const std::string &tag,
        const dims_t &strides_ = {});
int check_mem_size(const dnnl_memory_desc_t &md);
int check_mem_size(const_dnnl_primitive_desc_t const_pd);

memory_kind_ext_t str2memory_kind(const char *str);

float reorder_rescale_factor();

#endif
