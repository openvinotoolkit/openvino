/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

#ifndef TEST_GEMM_COMMON_H
#define TEST_GEMM_COMMON_H

#include <cstdint>
#include <utility>
#include <vector>
#include <type_traits>

#include "test_gemm_data_preparation.hpp"
#include "test_gemm_params.hpp"
#include "test_gemm_validation.hpp"

#include "dnnl_test_common.hpp"
#include "dnnl_thread.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_types.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "tests/test_thread.hpp"
#endif

#include "tests/test_isa_common.hpp"

#define CONCAT_WITH_UNDERSCORE_(a, b) a##_##b
#define CONCAT_WITH_UNDERSCORE(a, b) CONCAT_WITH_UNDERSCORE_(a, b)

#define INST_TEST_CASE_(str, ...) \
    INSTANTIATE_TEST_SUITE_P(str, gemm_test, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE(str, ...) \
    INST_TEST_CASE_( \
            CONCAT_WITH_UNDERSCORE(str, TEST_CASE_NAME_PREFIX), __VA_ARGS__)

#define CPU_INST_TEST_CASE_(str, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P(str, gemm_test, ::testing::Values(__VA_ARGS__))
#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INST_TEST_CASE_( \
            CONCAT_WITH_UNDERSCORE(str, TEST_CASE_NAME_PREFIX), __VA_ARGS__)

// Declare bfloat16 GEMM interfaces for testing
extern "C" {
dnnl_status_t dnnl_gemm_bf16bf16f32(char transa, char transb, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const bfloat16_t *A,
        dnnl_dim_t lda, const bfloat16_t *B, dnnl_dim_t ldb, float beta,
        float *C, dnnl_dim_t ldc);
}

// Declare packed GEMM interfaces for testing
#include "src/cpu/gemm/gemm_pack.hpp"

namespace dnnl {

#if defined(DNNL_WTIH_SYCL)
bool is_memory_kind_buffer(const test_memory &mem) {
    return sycl_interop::get_memory_kind(mem.get())
            == sycl_interop::memory_kind::buffer;
}
#endif

/* Test implementation description.
 * The testing steps looks as follows:
 * 0.  Prepare mapper_m and mapper_n <- details in test_gemm_data_preparation.hpp
 * 1.a Generate random matrices A', B', C'
 * 1.b Prepare matrices A, B, C based on A', B', and C' respectively
 * 2.  Compute C_calc := Op(M, N, K, A, B, C)
 * 3.  Compute C'_ref := Op_REF(M_test, N_test, K, A', B', C')
 * 4.  Expand C'_ref to C_ref, by applying mapper_m and mapper_n
 * 5.  Compare C_calc and C_ref
 */

template <typename a_dt, typename b_dt, typename c_dt>
struct dnnl_gemm {
    static dnnl_status_t call(test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem) {
        throw error(dnnl_runtime_error, "unknown gemm");
    }
};

template <>
struct dnnl_gemm<float16_t, float16_t, float16_t> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {
        throw error(dnnl_runtime_error, "unknown gemm");
    }
};

template <>
struct dnnl_gemm<float, float, float> {
    static dnnl_status_t call_packed(const test_params &p,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem) {
        /* Alas, the internal API still uses Fortran notation.
         * So in addition to the changes for pack API, we also need to take
         * care of conversions and layouts */

        using namespace dnnl::impl::cpu;

        assert(p.alpha == 1.f);

        /* Prepare for Fortran style, hence A <-> B */
        char trans_a = p.transB, trans_b = p.transA;

        int64_t m = p.N, n = p.M, k = p.K;
        int64_t lda = p.ldb, ldb = p.lda, ldc = p.ldc;

        std::vector<float> a_pack_buf, b_pack_buf;
        float *A = map_memory<float>(b_mem), *a_eff = A;
        float *B = map_memory<float>(a_mem), *b_eff = B;
        float *C = map_memory<float>(c_mem);

        bool pack_a = p.pack_params.pack_b;
        bool pack_b = p.pack_params.pack_a;

        dnnl_status_t status = dnnl_success;

        if (pack_a) {
            size_t a_sz;
            status = sgemm_pack_get_size("A", &trans_a, &trans_b, &m, &n, &k,
                    &lda, &ldb, &a_sz, &pack_a);
            if (status != dnnl_success) return status;

            if (pack_a) {
                a_pack_buf.resize(a_sz / sizeof(float));
                a_eff = a_pack_buf.data();

                status = sgemm_pack("A", &trans_a, &trans_b, &m, &n, &k, &lda,
                        &ldb, A, a_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_b) {
            size_t b_sz;
            status = sgemm_pack_get_size("B", &trans_a, &trans_b, &m, &n, &k,
                    &lda, &ldb, &b_sz, &pack_b);
            if (status != dnnl_success) return status;

            if (pack_b) {
                b_pack_buf.resize(b_sz / sizeof(float));
                b_eff = b_pack_buf.data();

                status = sgemm_pack("B", &trans_a, &trans_b, &m, &n, &k, &lda,
                        &ldb, B, b_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_a) trans_a = 'P';
        if (pack_b) trans_b = 'P';

        status = sgemm_compute(&trans_a, &trans_b, &m, &n, &k, a_eff, &lda,
                b_eff, &ldb, &p.beta, C, &ldc);

        return status;
    }

    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {

        if (p.pack_params.pack_a || p.pack_params.pack_b)
            return call_packed(p, a_mem, b_mem, c_mem);

        auto A = map_memory<float>(a_mem);
        auto B = map_memory<float>(b_mem);
        auto C = map_memory<float>(c_mem);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        static auto *st = dnnl::testing::get_threadpool();
        return static_cast<dnnl_status_t>(dnnl::threadpool_interop::sgemm(
                p.transA, p.transB, p.M, p.N, p.K, p.alpha, A, p.lda, B, p.ldb,
                p.beta, C, p.ldc, st));
#else
        return dnnl_sgemm(p.transA, p.transB, p.M, p.N, p.K, p.alpha, A, p.lda,
                B, p.ldb, p.beta, C, p.ldc);
#endif
    }
};

template <>
struct dnnl_gemm<int8_t, int8_t, int32_t> {
    static dnnl_status_t call_packed(const test_params &p,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem, const test_memory &oc_mem) {
        /* Alas, the internal API still uses Fortran notation.
         * So in addition to the changes for pack API, we also need to take
         * care of conversions and layouts */

        using namespace dnnl::impl::cpu;

        assert(p.alpha == 1.f);
        assert(p.igemm_params.oa() == 0);
        assert(p.igemm_params.ob() == 0);

        /* Prepare for Fortran style, hence A <-> B */
        char trans_a = p.transB, trans_b = p.transA;

        int64_t m = p.N, n = p.M, k = p.K;
        int64_t lda = p.ldb, ldb = p.lda, ldc = p.ldc;

        int8_t *A = map_memory<int8_t>(b_mem), *a_eff = A;
        int8_t *B = map_memory<int8_t>(a_mem), *b_eff = B;

        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);

        char offset_c = '\0';
        switch (p.igemm_params.offsetc) {
            case 'R': offset_c = 'C'; break;
            case 'r': offset_c = 'c'; break;
            case 'C': offset_c = 'R'; break;
            case 'c': offset_c = 'r'; break;
            default: offset_c = p.igemm_params.offsetc;
        }

        std::vector<int8_t> a_pack_buf;
        std::vector<int8_t> b_pack_buf;
        bool pack_a = p.pack_params.pack_b;
        bool pack_b = p.pack_params.pack_a;

        dnnl_status_t status = dnnl_success;

        if (pack_a) {
            size_t a_sz;
            status = gemm_s8s8s32_pack_get_size(
                    "A", &trans_a, &trans_b, &m, &n, &k, &lda, &ldb, &a_sz);
            if (status != dnnl_success) return status;

            if (pack_a) {
                a_pack_buf.resize(a_sz);
                a_eff = a_pack_buf.data();

                status = gemm_s8s8s32_pack("A", &trans_a, &trans_b, &m, &n, &k,
                        &lda, &ldb, A, a_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_b) {
            size_t b_sz;

            status = gemm_s8s8s32_pack_get_size(
                    "B", &trans_a, &trans_b, &m, &n, &k, &lda, &ldb, &b_sz);
            if (status != dnnl_success) return status;

            if (pack_b) {
                b_pack_buf.resize(b_sz);
                b_eff = b_pack_buf.data();

                status = gemm_s8s8s32_pack("B", &trans_a, &trans_b, &m, &n, &k,
                        &lda, &ldb, B, b_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_a) trans_a = 'P';
        if (pack_b) trans_b = 'P';

        status = gemm_s8s8s32_compute(&trans_a, &trans_b, &offset_c, &m, &n, &k,
                a_eff, &lda, b_eff, &ldb, &p.beta, C, &ldc, oc);

        return status;
    }

    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &oc_mem) {

        if (p.pack_params.pack_a || p.pack_params.pack_b)
            return call_packed(p, a_mem, b_mem, c_mem, oc_mem);

        auto A = map_memory<int8_t>(a_mem);
        auto B = map_memory<int8_t>(b_mem);
        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);
        int8_t oa = p.igemm_params.oa();
        int8_t ob = p.igemm_params.ob();
        return dnnl_gemm_s8s8s32(p.transA, p.transB, p.igemm_params.offsetc,
                p.M, p.N, p.K, p.alpha, A, p.lda, oa, B, p.ldb, ob, p.beta, C,
                p.ldc, oc);
    }
};

template <>
struct dnnl_gemm<int8_t, uint8_t, int32_t> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &oc_mem) {
        throw error(dnnl_runtime_error, "unknown gemm");
    }
};

template <>
struct dnnl_gemm<uint8_t, uint8_t, int32_t> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &oc_mem) {

        throw error(dnnl_runtime_error, "unknown gemm");
    }
};

template <>
struct dnnl_gemm<uint8_t, int8_t, int32_t> {
    static dnnl_status_t call_packed(const test_params &p,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem, const test_memory &oc_mem) {
        /* Alas, the internal API still uses Fortran notation.
         * So in addition to the changes for pack API, we also need to take
         * care of conversions and layouts */

        using namespace dnnl::impl::cpu;

        assert(p.alpha == 1.f);
        assert(p.igemm_params.oa() == 0);
        assert(p.igemm_params.ob() == 0);

        /* Prepare for Fortran style, hence A <-> B */
        char trans_a = p.transB, trans_b = p.transA;

        int64_t m = p.N, n = p.M, k = p.K;
        int64_t lda = p.ldb, ldb = p.lda, ldc = p.ldc;

        int8_t *A = map_memory<int8_t>(b_mem), *a_eff = A;
        uint8_t *B = map_memory<uint8_t>(a_mem), *b_eff = B;

        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);

        char offset_c = '\0';
        switch (p.igemm_params.offsetc) {
            case 'R': offset_c = 'C'; break;
            case 'r': offset_c = 'c'; break;
            case 'C': offset_c = 'R'; break;
            case 'c': offset_c = 'r'; break;
            default: offset_c = p.igemm_params.offsetc;
        }

        std::vector<int8_t> a_pack_buf;
        std::vector<uint8_t> b_pack_buf;
        bool pack_a = p.pack_params.pack_b;
        bool pack_b = p.pack_params.pack_a;

        dnnl_status_t status = dnnl_success;

        if (pack_a) {
            size_t a_sz;
            status = gemm_s8u8s32_pack_get_size(
                    "A", &trans_a, &trans_b, &m, &n, &k, &lda, &ldb, &a_sz);
            if (status != dnnl_success) return status;

            if (pack_a) {
                a_pack_buf.resize(a_sz);
                a_eff = a_pack_buf.data();

                status = gemm_s8u8s32_pack("A", &trans_a, &trans_b, &m, &n, &k,
                        &lda, &ldb, A, a_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_b) {
            size_t b_sz;

            status = gemm_s8u8s32_pack_get_size(
                    "B", &trans_a, &trans_b, &m, &n, &k, &lda, &ldb, &b_sz);
            if (status != dnnl_success) return status;

            if (pack_b) {
                b_pack_buf.resize(b_sz);
                b_eff = b_pack_buf.data();

                status = gemm_s8u8s32_pack("B", &trans_a, &trans_b, &m, &n, &k,
                        &lda, &ldb, B, b_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_a) trans_a = 'P';
        if (pack_b) trans_b = 'P';

        status = gemm_s8u8s32_compute(&trans_a, &trans_b, &offset_c, &m, &n, &k,
                a_eff, &lda, b_eff, &ldb, &p.beta, C, &ldc, oc);

        return status;
    }

    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &oc_mem) {
        assert(p.igemm_params.oa() >= 0);

        if (p.pack_params.pack_a || p.pack_params.pack_b)
            return call_packed(p, a_mem, b_mem, c_mem, oc_mem);

        auto A = map_memory<uint8_t>(a_mem);
        auto B = map_memory<int8_t>(b_mem);
        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);
        uint8_t oa = (uint8_t)p.igemm_params.oa();
        int8_t ob = p.igemm_params.ob();

        return dnnl_gemm_u8s8s32(p.transA, p.transB, p.igemm_params.offsetc,
                p.M, p.N, p.K, p.alpha, A, p.lda, oa, B, p.ldb, ob, p.beta, C,
                p.ldc, oc);
    }
};

template <>
struct dnnl_gemm<float16_t, float16_t, float> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {
        return dnnl_unimplemented;
    }
};

template <>
struct dnnl_gemm<bfloat16_t, bfloat16_t, float> {
    static dnnl_status_t call_packed(const test_params &p,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem) {
        /* Alas, the internal API still uses Fortran notation.
         * So in addition to the changes for pack API, we also need to take
         * care of conversions and layouts */

        using namespace dnnl::impl::cpu;

        assert(p.alpha == 1.f);

        /* Prepare for Fortran style, hence A <-> B */
        char trans_a = p.transB, trans_b = p.transA;

        int64_t m = p.N, n = p.M, k = p.K;
        int64_t lda = p.ldb, ldb = p.lda, ldc = p.ldc;

        std::vector<bfloat16_t> a_pack_buf, b_pack_buf;
        bfloat16_t *A = map_memory<bfloat16_t>(b_mem), *a_eff = A;
        bfloat16_t *B = map_memory<bfloat16_t>(a_mem), *b_eff = B;
        float *C = map_memory<float>(c_mem);

        bool pack_a = p.pack_params.pack_b;
        bool pack_b = p.pack_params.pack_a;

        dnnl_status_t status = dnnl_success;

        if (pack_a) {
            size_t a_sz;
            status = gemm_bf16bf16f32_pack_get_size("A", &trans_a, &trans_b, &m,
                    &n, &k, &lda, &ldb, &a_sz, &pack_a);
            if (status != dnnl_success) return status;

            if (pack_a) {
                a_pack_buf.resize(a_sz / sizeof(*a_eff));
                a_eff = a_pack_buf.data();

                status = gemm_bf16bf16f32_pack("A", &trans_a, &trans_b, &m, &n,
                        &k, &lda, &ldb, A, a_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_b) {
            size_t b_sz;
            status = gemm_bf16bf16f32_pack_get_size("B", &trans_a, &trans_b, &m,
                    &n, &k, &lda, &ldb, &b_sz, &pack_b);
            if (status != dnnl_success) return status;

            if (pack_b) {
                b_pack_buf.resize(b_sz / sizeof(*b_eff));
                b_eff = b_pack_buf.data();

                status = gemm_bf16bf16f32_pack("B", &trans_a, &trans_b, &m, &n,
                        &k, &lda, &ldb, B, b_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_a) trans_a = 'P';
        if (pack_b) trans_b = 'P';

        status = gemm_bf16bf16f32_compute(&trans_a, &trans_b, &m, &n, &k, a_eff,
                &lda, b_eff, &ldb, &p.beta, C, &ldc);

        return status;
    }

    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {
        if (p.pack_params.pack_a || p.pack_params.pack_b)
            return call_packed(p, a_mem, b_mem, c_mem);

        auto A = map_memory<bfloat16_t>(a_mem);
        auto B = map_memory<bfloat16_t>(b_mem);
        auto C = map_memory<float>(c_mem);
        return dnnl_gemm_bf16bf16f32(p.transA, p.transB, p.M, p.N, p.K, p.alpha,
                A, p.lda, B, p.ldb, p.beta, C, p.ldc);
    }
};

template <>
struct dnnl_gemm<bfloat16_t, bfloat16_t, bfloat16_t> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {
        return dnnl_unimplemented;
    }
};

template <typename a_dt, typename b_dt, typename c_dt>
struct run_test_gemm {
    static void call(const test_params &p) {
        if (p.expect_to_fail) {
            engine eng = get_test_engine();
            test_memory zero_mem({}, eng);
            auto status = dnnl_gemm<a_dt, b_dt, c_dt>::call(
                    p, zero_mem, zero_mem, zero_mem, zero_mem);
            if (status != dnnl_success)
                throw error(status, "oneDNN gemm returned error");
            return;
        }

        test_gemm_data gemm_data;
        prepare_data_for_gemm_testing<a_dt, b_dt, c_dt>(p, gemm_data);

        auto status = dnnl_gemm<a_dt, b_dt, c_dt>::call(p, *gemm_data.a_mem,
                *gemm_data.b_mem, *gemm_data.c_mem, *gemm_data.oc_mem);

        if (status == dnnl_success) {
            validate<a_dt, b_dt, c_dt>(p, gemm_data);
        }

        if (status != dnnl_success)
            throw error(status, "oneDNN gemm returned error");
    }
};

template <typename a_dt, typename b_dt, typename c_dt>
class gemm_test_common : public ::testing::TestWithParam<test_params> {
protected:
    virtual void SetUp() {
        const auto &p = ::testing::TestWithParam<test_params>::GetParam();

        SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
                "GPU GEMM not implemented.");

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        SKIP_IF(get_test_engine_kind() == engine::kind::cpu,
                "SYCL CPU GEMM not implemented.");
#endif

        bool zero_off = (p.off.a == 0 && p.off.b == 0 && p.off.c == 0);
        SKIP_IF(!zero_off && get_test_engine_kind() == engine::kind::cpu,
                "CPU does not support non-zero offsets.");

        SKIP_IF(unsupported_data_type(data_traits<a_dt>::data_type),
                "Engine does not support this data type.");

        bool is_f16 = (data_traits<a_dt>::data_type == memory::data_type::f16);
        SKIP_IF(is_f16 && get_test_engine_kind() == engine::kind::cpu,
                "CPU does not support f16 data type.");

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        SKIP_IF(get_test_engine_kind() == engine::kind::cpu,
                "SYCL CPU GEMM not implemented.");
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        SKIP_IF(get_test_engine_kind() == engine::kind::gpu
                        && (data_traits<a_dt>::data_type
                                        == memory::data_type::u8
                                || data_traits<a_dt>::data_type
                                        == memory::data_type::s8),
                "SYCL GPU int GEMM not implemented.");
        SKIP_IF_CUDA(true, "Test not supported in CUDA backend");
#endif

#if DNNL_X64
        bool is_bf16bf16f32 = true
                && data_traits<a_dt>::data_type == memory::data_type::bf16
                && data_traits<b_dt>::data_type == memory::data_type::bf16
                && data_traits<c_dt>::data_type == memory::data_type::f32;

        SKIP_IF(is_bf16bf16f32 && get_test_engine_kind() == engine::kind::cpu
                        && !dnnl::mayiuse(cpu_isa::avx512_core),
                "Skip test for systems that do not support avx512_core.");
#endif

        bool pack = (p.pack_params.pack_a || p.pack_params.pack_b);
        SKIP_IF(!DNNL_X64 && pack,
                "Packed GEMM does not support non-x64 CPUs.");
        SKIP_IF((p.alpha != 1.f || p.igemm_params.oa() != 0
                        || p.igemm_params.ob() != 0)
                        && pack,
                "Packed GEMM doesn't support alpha or non-zero offset{A,B}.");
        SKIP_IF(data_traits<b_dt>::data_type == memory::data_type::u8
                        && get_test_engine_kind() == engine::kind::cpu,
                "CPU does not support s8u8s32 and u8u8s32 GEMM.");
        SKIP_IF(data_traits<c_dt>::data_type == memory::data_type::bf16
                        && get_test_engine_kind() == engine::kind::cpu,
                "CPU does not support bf16bf16bf16 GEMM.");

        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status, false);
    }
    void Test() {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
        testing::scoped_tp_activation_t sta;
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            const auto &p = ::testing::TestWithParam<test_params>::GetParam();

#if defined(TEST_DNNL_DPCPP_BUFFER)
            // Test SYCL buffer interfaces
            run_test_gemm<a_dt, b_dt, c_dt>::call(p);
#else
            // Test SYCL USM interfaces
            bool zero_off = (p.off.a == 0 && p.off.b == 0 && p.off.c == 0);
            SKIP_IF(!zero_off, "USM interfaces do not support offsets.");

            run_test_gemm<a_dt, b_dt, c_dt>::call(p);
#endif

            return;
        }
#endif
        const auto &p = ::testing::TestWithParam<test_params>::GetParam();
        run_test_gemm<a_dt, b_dt, c_dt>::call(p);
    }
};
} // namespace dnnl
#endif
