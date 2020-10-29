/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "mkldnn_thread.hpp"
#include "mkldnn_traits.hpp"
#include "mkldnn_types.h"

#include "gemm_pack.hpp"

#include "cpu_isa_traits.hpp"

#include "gemm.hpp"
#include "gemm_driver.hpp"
#include "os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#if USE_MKL_PACKED_GEMM
static inline CBLAS_IDENTIFIER cblas_identifier(const char *identifier) {
    return utils::one_of(*identifier, 'a', 'A') ? CblasAMatrix : CblasBMatrix;
}

static inline CBLAS_TRANSPOSE cblas_transpose(const char *trans) {
    return utils::one_of(*trans, 'n', 'N') ? CblasNoTrans : CblasTrans;
}

static inline MKL_INT cblas_storage(const char *trans) {
    switch (*trans) {
        case 'N':
        case 'n': return CblasNoTrans;
        case 'T':
        case 't': return CblasTrans;
        default: return CblasPacked;
    }
}

static inline CBLAS_OFFSET cblas_offset(const char *offset) {
    switch (*offset) {
        case 'R':
        case 'r': return CblasRowOffset;
        case 'C':
        case 'c': return CblasColOffset;
        default: return CblasFixOffset;
    }
}
#endif

#if !USE_MKL_PACKED_GEMM
template <typename a_dt, typename b_dt>
static inline bool use_reference_igemm(void) {
    constexpr bool is_s8u8 = true
            && data_traits<a_dt>::data_type == data_type::s8
            && data_traits<b_dt>::data_type == data_type::u8;
    if (is_s8u8)
        return !mayiuse(sse42) || mayiuse(avx512_mic);
    else
        return !mayiuse(avx512_core);
}

template <typename T>
static bool is_good_ld(dim_t ld) {
    static constexpr auto align = 64 / sizeof(T);
    static constexpr auto no_align = 2048 / sizeof(T);

    return ((ld % align) == 0) && ((ld % no_align) != 0);
}
#else
template <typename a_dt, typename b_dt>
static inline bool use_reference_igemm(void) {
    return false;
}

template <typename T>
static bool is_good_ld(dim_t ld) {
    return false;
}
#endif

static mkldnn_status_t check_pack_get_size_input(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb) {

    if (utils::any_null(identifier, transa, transb, M, N, K, lda, ldb))
        return mkldnn_invalid_arguments;

    bool is_transa = utils::one_of(*transa, 'T', 't');
    bool is_transb = utils::one_of(*transb, 'T', 't');

    bool ok = true && utils::one_of(*transa, 'T', 't', 'N', 'n')
            && utils::one_of(*transb, 'T', 't', 'N', 'n')
            && utils::one_of(*identifier, 'A', 'a', 'B', 'b') && *M >= 0
            && *N >= 0 && *K >= 0 && *lda >= nstl::max(1, !is_transa ? *M : *K)
            && *ldb >= nstl::max(1, !is_transb ? *K : *N);

    if (!ok) return mkldnn_invalid_arguments;

    return mkldnn_success;
}

static mkldnn_status_t check_pack_input(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const float *alpha, const int *lda, const int *ldb,
        const void *src, void *dst) {
    if (utils::any_null(src, dst, alpha)) return mkldnn_invalid_arguments;

    return check_pack_get_size_input(
            identifier, transa, transb, M, N, K, lda, ldb);
}

template <typename a_dt, typename b_dt, typename c_dt>
static mkldnn_status_t gemm_pack_driver(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const float *alpha, const int *lda, const int *ldb,
        const void *src, gemm_pack_storage_t *pack_dst, bool measure_only) {

    a_dt oa = 0;
    b_dt ob = 0;

    const a_dt *a = nullptr;
    const b_dt *b = nullptr;
    pack_type packing;

    if (utils::one_of(*identifier, 'a', 'A')) {
        a = (const a_dt *)src;
        packing = pack_type::pack_a;
    } else {
        b = (const b_dt *)src;
        packing = pack_type::pack_b;
    }

    return gemm_driver<a_dt, b_dt, c_dt>(transa, transb, "N", M, N, K, alpha, a,
            lda, &oa, b, ldb, &ob, nullptr, nullptr, nullptr, nullptr, false,
            packing, pack_dst, measure_only);
}

mkldnn_status_t sgemm_pack_get_size(const char *identifier, const char *transa,
        const char *transb, const int *M, const int *N, const int *K,
        const int *lda, const int *ldb, size_t *size, bool *pack) {

    if (!pack_sgemm_supported()) return mkldnn_unimplemented;

    mkldnn_status_t result;
    *size = 0;
    if (pack) *pack = true;

    result = check_pack_get_size_input(
            identifier, transa, transb, M, N, K, lda, ldb);
    if (result != mkldnn_success) return result;

#if USE_MKL_PACKED_GEMM
    *size = cblas_sgemm_pack_get_size(cblas_identifier(identifier), *M, *N, *K);
#else
    bool do_a = utils::one_of(*identifier, 'a', 'A');
    float alpha = 1.0f;
    gemm_pack_storage_shell_t shell {mkldnn_get_max_threads()};

    result = gemm_pack_driver<float, float, float>(identifier, transa, transb,
            M, N, K, &alpha, lda, ldb, nullptr, &shell, true);
    if (result != mkldnn_success) return result;

    *size = shell.size();
    if (pack) {
        *pack = !(shell.single_nocopy()
                && utils::one_of(do_a ? *transa : *transb, 'n', 'N')
                && is_good_ld<float>(do_a ? *lda : *ldb));
    }
#endif

    return mkldnn_success;
}

mkldnn_status_t gemm_bf16bf16f32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack) {

    if (!pack_gemm_bf16bf16f32_supported()) return mkldnn_unimplemented;

    mkldnn_status_t result;
    *size = 0;
    if (pack) *pack = true;

    int M_s32 = (int)*M;
    int N_s32 = (int)*N;
    int K_s32 = (int)*K;
    int lda_s32 = (int)*lda;
    int ldb_s32 = (int)*ldb;

    result = check_pack_get_size_input(identifier, transa, transb, &M_s32,
            &N_s32, &K_s32, &lda_s32, &ldb_s32);
    if (result != mkldnn_success) return result;

    float alpha = 1.0f;
    gemm_pack_storage_shell_t shell {mkldnn_get_max_threads()};

    result = gemm_pack_driver<mkldnn_bfloat16_t, mkldnn_bfloat16_t, float>(identifier, transa,
            transb, &M_s32, &N_s32, &K_s32, &alpha, &lda_s32, &ldb_s32, nullptr,
            &shell, true);
    if (result != mkldnn_success) return result;

    *size = shell.size();

    return mkldnn_success;
}

template <typename a_dt, typename b_dt>
mkldnn_status_t gemm_x8x8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack) {

    mkldnn_status_t result;
    *size = 0;
    if (pack) *pack = true;

    result = check_pack_get_size_input(
            identifier, transa, transb, M, N, K, lda, ldb);
    if (result != mkldnn_success) return result;

#if USE_MKL_PACKED_GEMM
    constexpr bool is_s8u8 = true
            && data_traits<a_dt>::data_type == data_type::s8
            && data_traits<b_dt>::data_type == data_type::u8;

    if (is_s8u8) {
        *size = cblas_gemm_s8u8s32_pack_get_size(
                cblas_identifier(identifier), *M, *N, *K);
        return mkldnn_success;
    }
#endif

    bool do_a = utils::one_of(*identifier, 'a', 'A');
    float alpha = 1.0f;
    gemm_pack_storage_shell_t shell {mkldnn_get_max_threads(), do_a, !do_a};

    if (!use_reference_igemm<a_dt, b_dt>()) {
        result = gemm_pack_driver<a_dt, b_dt, int32_t>(identifier, transa,
                transb, M, N, K, &alpha, lda, ldb, nullptr, &shell, true);
        if (result != mkldnn_success) return result;
    } else {
        auto rows = do_a ? *M : *K;
        auto cols = do_a ? *K : *N;
        prep_ref_gemm_s8u8s32_pack(do_a, rows, cols, &shell);
    }

    *size = shell.size();
    if (pack) {
        *pack = !(shell.single_nocopy()
                && utils::one_of(do_a ? *transa : *transb, 'n', 'N')
                && is_good_ld<float>(do_a ? *lda : *ldb));
    }

    return mkldnn_success;
}

mkldnn_status_t gemm_s8u8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack) {

    return gemm_x8x8s32_pack_get_size<int8_t, uint8_t>(
            identifier, transa, transb, M, N, K, lda, ldb, size, pack);
}

mkldnn_status_t gemm_s8s8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack) {

    return gemm_x8x8s32_pack_get_size<int8_t, int8_t>(
            identifier, transa, transb, M, N, K, lda, ldb, size, pack);
}

mkldnn_status_t sgemm_pack(const char *identifier, const char *transa,
        const char *transb, const int *M, const int *N, const int *K,
        const int *lda, const int *ldb, const float *src, float *dst) {
    float one = 1.f, *alpha = &one;

    if (!pack_sgemm_supported()) return mkldnn_unimplemented;

    auto result = check_pack_input(
            identifier, transa, transb, M, N, K, alpha, lda, ldb, src, dst);
    if (result != mkldnn_success) return result;

#if USE_MKL_PACKED_GEMM
    auto cblas_id = cblas_identifier(identifier);
    auto ld = (cblas_id == CblasAMatrix) ? *lda : *ldb;
    auto trans = (cblas_id == CblasAMatrix) ? transa : transb;
    cblas_sgemm_pack(CblasColMajor, cblas_id, cblas_transpose(trans), *M, *N,
            *K, *alpha, src, ld, dst);
    return mkldnn_success;
#else
    gemm_pack_storage_t pack_dst {dst};

    return gemm_pack_driver<float, float, float>(identifier, transa, transb, M,
            N, K, alpha, lda, ldb, src, &pack_dst, false);
#endif
}

mkldnn_status_t gemm_bf16bf16f32_pack(const char *identifier, const char *transa,
        const char *transb, const int *M, const int *N, const int *K,
        const int *lda, const int *ldb, const mkldnn_bfloat16_t *src,
        mkldnn_bfloat16_t *dst) {
    float one = 1.f, *alpha = &one;

    if (!pack_gemm_bf16bf16f32_supported()) return mkldnn_unimplemented;

    int M_s32 = (int)*M;
    int N_s32 = (int)*N;
    int K_s32 = (int)*K;
    int lda_s32 = (int)*lda;
    int ldb_s32 = (int)*ldb;

    auto result = check_pack_input(identifier, transa, transb, &M_s32, &N_s32,
            &K_s32, alpha, &lda_s32, &ldb_s32, src, dst);
    if (result != mkldnn_success) return result;

    gemm_pack_storage_t pack_dst {dst};

    return gemm_pack_driver<mkldnn_bfloat16_t, mkldnn_bfloat16_t, float>(identifier, transa,
            transb, &M_s32, &N_s32, &K_s32, alpha, &lda_s32, &ldb_s32, src,
            &pack_dst, false);
}

template <typename a_dt, typename b_dt>
mkldnn_status_t gemm_x8x8s32_pack(const char *identifier, const char *transa,
        const char *transb, const int *M, const int *N, const int *K,
        const int *lda, const int *ldb, const void *src, void *dst) {

    float alpha = 1.0f; // Not used with igemm.
    auto result = check_pack_input(
            identifier, transa, transb, M, N, K, &alpha, lda, ldb, src, dst);
    if (result != mkldnn_success) return result;

#if USE_MKL_PACKED_GEMM
    constexpr bool is_s8u8 = true
            && data_traits<a_dt>::data_type == data_type::s8
            && data_traits<b_dt>::data_type == data_type::u8;

    if (is_s8u8) {
        auto cblas_id = cblas_identifier(identifier);
        auto ld = (cblas_id == CblasAMatrix) ? *lda : *ldb;
        auto trans = (cblas_id == CblasAMatrix) ? transa : transb;
        cblas_gemm_s8u8s32_pack(CblasColMajor, cblas_id, cblas_transpose(trans),
                *M, *N, *K, src, ld, dst);
        return mkldnn_success;
    }
#endif
    gemm_pack_storage_t pack_dst {dst};

    if (!use_reference_igemm<a_dt, b_dt>()) {
        return gemm_pack_driver<a_dt, b_dt, int32_t>(identifier, transa, transb,
                M, N, K, &alpha, lda, ldb, src, &pack_dst, false);
    } else {
        bool do_a = utils::one_of(*identifier, 'a', 'A');
        bool is_trans = utils::one_of(do_a ? *transa : *transb, 't', 'T');
        auto ld = do_a ? *lda : *ldb;
        auto rows = do_a ? *M : *K;
        auto cols = do_a ? *K : *N;

        prep_ref_gemm_s8u8s32_pack(do_a, rows, cols, &pack_dst);
        return ref_gemm_s8u8s32_pack(src, ld, rows, cols, is_trans, &pack_dst);
    }
}

mkldnn_status_t gemm_s8u8s32_pack(const char *identifier, const char *transa,
        const char *transb, const int *M, const int *N, const int *K,
        const int *lda, const int *ldb, const void *src, void *dst) {

    return gemm_x8x8s32_pack<int8_t, uint8_t>(
            identifier, transa, transb, M, N, K, lda, ldb, src, dst);
}

mkldnn_status_t gemm_s8s8s32_pack(const char *identifier, const char *transa,
        const char *transb, const int *M, const int *N, const int *K,
        const int *lda, const int *ldb, const void *src, void *dst) {

    return gemm_x8x8s32_pack<int8_t, int8_t>(
            identifier, transa, transb, M, N, K, lda, ldb, src, dst);
}

mkldnn_status_t sgemm_compute(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *A,
        const int *lda, const float *B, const int *ldb, const float *beta,
        float *C, const int *ldc) {

#if USE_MKL_PACKED_GEMM
    if (utils::any_null(transa, transb, M, N, K, A, lda, B, ldb, beta, C, ldc))
        return mkldnn_invalid_arguments;
    cblas_sgemm_compute(CblasColMajor, cblas_storage(transa),
            cblas_storage(transb), *M, *N, *K, A, *lda, B, *ldb, *beta, C,
            *ldc);
    return mkldnn_success;
#else
    if (!pack_sgemm_supported()) return mkldnn_unimplemented;

    float one = 1.0f;

    return extended_sgemm(
            transa, transb, M, N, K, &one, A, lda, B, ldb, beta, C, ldc);
#endif
}

mkldnn_status_t gemm_bf16bf16f32_compute(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const mkldnn_bfloat16_t *A,
        const int *lda, const mkldnn_bfloat16_t *B, const int *ldb, const float *beta,
        float *C, const int *ldc) {

    if (!pack_gemm_bf16bf16f32_supported()) return mkldnn_unimplemented;

    float one = 1.0f;

    return gemm_bf16bf16f32(
            transa, transb, M, N, K, &one, A, lda, B, ldb, beta, C, ldc);
}

template <typename a_dt, typename b_dt>
mkldnn_status_t gemm_x8x8s32_compute(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const a_dt *A, const int *lda, const b_dt *B, const int *ldb,
        const float *beta, int32_t *C, const int *ldc, const int32_t *co) {

    const float one = 1.f, *alpha = &one;
    const a_dt zero_a_dt = 0, *ao = &zero_a_dt;
    const b_dt zero_b_dt = 0, *bo = &zero_b_dt;

#if USE_MKL_PACKED_GEMM
    constexpr bool is_s8u8 = true
            && data_traits<a_dt>::data_type == data_type::s8
            && data_traits<b_dt>::data_type == data_type::u8;

    if (is_s8u8) {
        if (utils::any_null(transa, transb, offsetc, M, N, K, alpha, A, lda, ao,
                    B, ldb, bo, beta, C, ldc, co))
            return mkldnn_invalid_arguments;
        cblas_gemm_s8u8s32_compute(CblasColMajor, cblas_storage(transa),
                cblas_storage(transb), cblas_offset(offsetc), *M, *N, *K,
                *alpha, A, *lda, *ao, B, *ldb, *bo, *beta, C, *ldc, co);
        return mkldnn_success;
    }
#endif
    auto lda_eff = *lda, ldb_eff = *ldb;
    auto transa_eff = *transa, transb_eff = *transb;

    if (!use_reference_igemm<a_dt, b_dt>()) {
        return gemm_s8x8s32(&transa_eff, &transb_eff, offsetc, M, N, K, alpha,
                A, &lda_eff, ao, B, &ldb_eff, bo, beta, C, ldc, co);
    } else {
        dim_t ld, td;

        if (transa_eff == 'p' || transa_eff == 'P') {
            gemm_pack_storage_t a_packed {A};
            if (!a_packed.get_nocopy(ld, td)) return mkldnn_invalid_arguments;
            A = a_packed.matrix<a_dt>();
            lda_eff = ld;
            transa_eff = 'N';
        }

        if (transb_eff == 'p' || transb_eff == 'P') {
            gemm_pack_storage_t b_packed {B};
            if (!b_packed.get_nocopy(ld, td)) return mkldnn_invalid_arguments;
            B = b_packed.matrix<b_dt>();
            ldb_eff = ld;
            transb_eff = 'N';
        }

        return gemm_s8x8s32(&transa_eff, &transb_eff, offsetc, M, N, K, alpha,
                A, &lda_eff, ao, B, &ldb_eff, bo, beta, C, ldc, co);
    }
}

mkldnn_status_t gemm_s8u8s32_compute(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const int8_t *A, const int *lda, const uint8_t *B, const int *ldb,
        const float *beta, int32_t *C, const int *ldc, const int32_t *co) {

    return gemm_x8x8s32_compute(
            transa, transb, offsetc, M, N, K, A, lda, B, ldb, beta, C, ldc, co);
}

mkldnn_status_t gemm_s8s8s32_compute(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const int8_t *A, const int *lda, const int8_t *B, const int *ldb,
        const float *beta, int32_t *C, const int *ldc, const int32_t *co) {

    return gemm_x8x8s32_compute(
            transa, transb, offsetc, M, N, K, A, lda, B, ldb, beta, C, ldc, co);
}

} // namespace cpu
} // namespace impl
} // namespace mkldnn
