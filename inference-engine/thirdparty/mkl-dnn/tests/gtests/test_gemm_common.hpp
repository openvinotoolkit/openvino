/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn_types.h"
#include "mkldnn.h"

#define CONCAT_WITH_UNDERSCORE_(a,b) a ## _ ## b
#define CONCAT_WITH_UNDERSCORE(a,b) CONCAT_WITH_UNDERSCORE_(a,b)

#define INST_TEST_CASE_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, gemm_test, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE(str, ...) INST_TEST_CASE_( \
        CONCAT_WITH_UNDERSCORE(str,TEST_CASE_NAME_PREFIX), __VA_ARGS__)

namespace mkldnn {

struct test_igemm_params {
    char offsetc;
    bool zero_oa;
    bool zero_ob;
    bool zero_oc;
};

struct test_params {
    char transA;
    char transB;
    int M;
    int N;
    int K;
    float alpha;
    float beta;
    int lda;
    int ldb;
    int ldc;

    test_igemm_params igemm_params;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t>
void ref_gemm(const char *transa, const char *transb, int m, int n, int k,
        const data_t alpha, const data_t *a, int lda, const data_t *b,
        int ldb, data_t beta, data_t *c, int ldc) {

    const bool tr_a = transa && (*transa == 'T' || *transa == 't');
    const bool tr_b = transb && (*transb == 'T' || *transb == 't');

    auto pa = [=] (int i, int j) { return a[j*lda + i]; };
    auto pb = [=] (int i, int j) { return b[j*ldb + i]; };
    auto pc = [=] (int i, int j) { return c[j*ldc + i]; };

    mkldnn::impl::parallel_nd(m, n, [&](int im, int in) {
        data_t c_elem = (beta == 0.) ? 0. : pc(im, in) * beta;

        for (int ik = 0; ik < k; ik++) {
            const data_t a_elem = tr_a ? pa(ik, im) : pa(im, ik);
            const data_t b_elem = tr_b ? pb(in, ik) : pb(ik, in);
            c_elem += alpha * a_elem * b_elem;
        }
        c[in*ldc + im] = c_elem;
    });
}

template <typename b_dt>
void ref_gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, int m, int n, int k, const float alpha,
        int8_t *A, int lda, const int8_t *oa, b_dt *B, int ldb,
        const int8_t *ob, const float beta, int32_t *C, int ldc,
        const int32_t *oc) {

    bool OCisR = (*offsetc == 'R' || *offsetc == 'r');
    bool OCisC = (*offsetc == 'C' || *offsetc == 'c');
    bool AisN = (*transa == 'N' || *transa == 'n');
    bool BisN = (*transb == 'N' || *transb == 'n');

    size_t sizeA = AisN ? lda * k : lda * m;
    size_t sizeB = BisN ? ldb * n : ldb * k;
    size_t sizeC = ldc * n;

    double *dA = (double *)test_malloc(sizeA * sizeof(double));
    double *dB = (double *)test_malloc(sizeB * sizeof(double));
    double *dC = (double *)test_malloc(sizeC * sizeof(double));

    auto da_setter = [=] (int i, int j, double v) { dA[j * lda + i] = v; };
    auto db_setter = [=] (int i, int j, double v) { dB[j * ldb + i] = v; };

    auto ia_accessor = [=] (int i, int j) { return A[j * lda + i]; };
    auto ib_accessor = [=] (int i, int j) { return B[j * ldb + i]; };

    const int a_rows = AisN ? m : k;
    const int a_cols = AisN ? k : m;
    mkldnn::impl::parallel_nd(a_cols, a_rows, [&](int j, int i) {
        da_setter(i, j,
            static_cast<double>(ia_accessor(i, j)) + static_cast<double>(oa[0]));
    });

    const int b_rows = BisN ? k : n;
    const int b_cols = BisN ? n : k;
    mkldnn::impl::parallel_nd(b_cols, b_rows, [&](int j, int i) {
        db_setter(i, j,
            static_cast<double>(ib_accessor(i, j)) + static_cast<double>(ob[0]));
    });

    ref_gemm(transa, transb, m, n, k, 1.0, dA, lda, dB, ldb, 0.0, dC, ldc);

    auto i2d = [=] (int32_t v) { return static_cast<double>(v); };
    auto f2d = [=] (float v) { return static_cast<double>(v); };

    mkldnn::impl::parallel_nd(n, m, [&] (int j, int i) {
        double coffset = OCisR ? i2d(oc[j]) : OCisC ? i2d(oc[i]) : i2d(oc[0]);
        double val = ((beta == 0.0f) ? 0.0 : f2d(beta) * i2d(C[i + j * ldc]))
            + f2d(alpha) * dC[i + j * ldc] + coffset;
        C[i + j * ldc] =
            static_cast<int32_t>(nearbyint(saturate<int32_t, double>(val)));
    });

    test_free((char *)dA);
    test_free((char *)dB);
    test_free((char *)dC);
}

template <typename b_dt, typename c_dt>
void compare(int m, int n, const c_dt *c, const c_dt *c_ref, int ldc,
        float alpha = 1.0f, float beta = 0.0f, int k = 1) {
    using data_type = memory::data_type;
    mkldnn::impl::parallel_nd(n, ldc, [&](int i, int j) {
        c_dt ref = c_ref[i*ldc + j];
        c_dt got = c[i*ldc + j];
        c_dt diff = got - ref;

        if (data_traits<b_dt>::data_type == data_type::f32) {
            c_dt e = (std::abs(ref) > 1e-4) ? diff / ref : diff;
            EXPECT_NEAR(e, 0.0, 1e-4) << "Row: " << j << " Col: " << i;
        } else {
            // igemm
            if (alpha == 1.0f) {
                EXPECT_NEAR(diff, 0, 1) << "Row: " << j << " Col: " << i;
            } else {
                if (data_traits<b_dt>::data_type == data_type::u8) {
                    c_dt eps = k / 1000 + 1;
                    EXPECT_NEAR(diff, 0, eps) << "Row: " << j << " Col: " << i;
                } else if (data_traits<b_dt>::data_type == data_type::s8) {
                    c_dt eps = k / 500 + 1;
                    EXPECT_NEAR(diff, 0, eps) << "Row: " << j << " Col: " << i;
                }
            }
        }
    });
}

inline void get_matrix_size(const test_params &p, size_t &sizeA,
        size_t &sizeB, size_t &sizeC) {
    const bool tr_a = (p.transA == 'T' || p.transA == 't');
    const bool tr_b = (p.transB == 'T' || p.transB == 't');
    sizeA = !tr_a ? p.lda * p.K : p.lda * p.M,
    sizeB = !tr_b ? p.ldb * p.N : p.ldb * p.K,
    sizeC = p.ldc * p.N;
}

template <typename T>
inline T* get_matrix_buffer(size_t n) {
    return (T*)test_malloc(n * sizeof(T));
}

template <typename a_dt, typename b_dt, typename c_dt>
inline void fill_matrix(const test_params &p, size_t sizeA, size_t sizeB,
        size_t sizeC, size_t sizeco, a_dt *A, b_dt *B, c_dt *C, a_dt *oa,
        a_dt *ob, c_dt *oc) {
    fill_data<a_dt>(sizeA, A);
    fill_data<b_dt>(sizeB, B);
    fill_data<c_dt>(sizeC, C);
    if (oa != nullptr && ob != nullptr && oc != nullptr) {
        if (p.igemm_params.zero_oa) (*oa) = 0;
        else fill_data<a_dt>(1, oa);

        if (p.igemm_params.zero_ob) (*ob) = 0;
        else fill_data<a_dt>(1, ob);

        if (p.igemm_params.zero_oc) {
            for (size_t i = 0; i < sizeco; i++)
                oc[i] = 0;
        } else fill_data<c_dt>(sizeco, oc);
    }
}

template <typename a_dt, typename b_dt, typename c_dt>
void run_test_gemm(const test_params &p) {}

template <>
void run_test_gemm<int8_t, uint8_t, int32_t>(const test_params &p) {
    size_t sizeA, sizeB, sizeC;
    get_matrix_size(p, sizeA, sizeB, sizeC);

    int8_t  *A = get_matrix_buffer<int8_t>(sizeA);
    uint8_t *B = get_matrix_buffer<uint8_t>(sizeB);
    int32_t *C = get_matrix_buffer<int32_t>(sizeC);
    int32_t *C_ref = get_matrix_buffer<int32_t>(sizeC);

    bool OCisR = (p.igemm_params.offsetc == 'R' || p.igemm_params.offsetc == 'r');
    bool OCisC = (p.igemm_params.offsetc == 'C' || p.igemm_params.offsetc == 'c');
    size_t sizeco = OCisR ? p.N : OCisC ? p.M : 1;

    int8_t oa, ob;
    int32_t *oc = get_matrix_buffer<int32_t>(sizeco);

    fill_matrix<int8_t, uint8_t, int32_t>(p, sizeA, sizeB, sizeC, sizeco,
        A, B, C, &oa, &ob, oc);

    mkldnn::impl::parallel_nd(p.ldc * p.N,
        [&](int i) { C_ref[i] = static_cast<int32_t>(C[i]); });

    auto status = mkldnn_gemm_s8u8s32(&p.transA, &p.transB, &p.igemm_params.offsetc,
        &p.M, &p.N, &p.K, &p.alpha, A, &p.lda, &oa, B, &p.ldb, &ob,
        &p.beta, C, &p.ldc, oc);

    if (status != mkldnn_success)
        throw error(status, "mkldnn_gemm_s8u8s32 returned error");

    ref_gemm_s8x8s32<uint8_t>(&p.transA, &p.transB, &p.igemm_params.offsetc, p.M, p.N,
        p.K, p.alpha, A, p.lda, &oa, B, p.ldb, &ob, p.beta, C_ref,
        p.ldc, oc);

    compare<uint8_t, int32_t>(p.M, p.N, C, C_ref, p.ldc, p.alpha, p.beta, p.K);

    test_free((char *)A);
    test_free((char *)B);
    test_free((char *)C);
    test_free((char *)C_ref);
    test_free((char *)oc);
}

template <>
void run_test_gemm<int8_t, int8_t, int32_t>(const test_params &p) {
    size_t sizeA, sizeB, sizeC;
    get_matrix_size(p, sizeA, sizeB, sizeC);

    int8_t  *A = get_matrix_buffer<int8_t>(sizeA);
    int8_t  *B = get_matrix_buffer<int8_t>(sizeB);
    int32_t *C = get_matrix_buffer<int32_t>(sizeC);
    int32_t *C_ref = get_matrix_buffer<int32_t>(sizeC);

    bool OCisR = (p.igemm_params.offsetc == 'R' || p.igemm_params.offsetc == 'r');
    bool OCisC = (p.igemm_params.offsetc == 'C' || p.igemm_params.offsetc == 'c');
    size_t sizeco = OCisR ? p.N : OCisC ? p.M : 1;

    int8_t oa, ob;
    int32_t* oc = get_matrix_buffer<int32_t>(sizeco);

    fill_matrix<int8_t, int8_t, int32_t>(p, sizeA, sizeB, sizeC, sizeco, A, B, C,
        &oa, &ob, oc);

    mkldnn::impl::parallel_nd(p.ldc * p.N,
        [&](int i) { C_ref[i] = static_cast<int32_t>(C[i]); });

    auto status = mkldnn_gemm_s8s8s32(&p.transA, &p.transB, &p.igemm_params.offsetc,
        &p.M, &p.N, &p.K, &p.alpha, A, &p.lda, &oa, B, &p.ldb, &ob,
        &p.beta, C, &p.ldc, oc);

    if (status != mkldnn_success)
        throw error(status, "mkldnn_gemm_s8s8s32 returned error");

    ref_gemm_s8x8s32<int8_t>(&p.transA, &p.transB, &p.igemm_params.offsetc, p.M, p.N,
        p.K, p.alpha, A, p.lda, &oa, B, p.ldb, &ob, p.beta, C_ref,
        p.ldc, oc);

    compare<int8_t, int32_t>(p.M, p.N, C, C_ref, p.ldc, p.alpha, p.beta, p.K);

    test_free((char *)A);
    test_free((char *)B);
    test_free((char *)C);
    test_free((char *)C_ref);
    test_free((char *)oc);
}

template <>
void run_test_gemm<float, float, float>(const test_params &p) {
    size_t sizeA, sizeB, sizeC;
    get_matrix_size(p, sizeA, sizeB, sizeC);

    float *A = get_matrix_buffer<float>(sizeA);
    float *B = get_matrix_buffer<float>(sizeB);
    float *C = get_matrix_buffer<float>(sizeC);
    float *C_ref = get_matrix_buffer<float>(sizeC);

    fill_matrix<float, float, float>(p, sizeA, sizeB, sizeC, 0, A, B, C,
        nullptr, nullptr, nullptr);

    mkldnn::impl::parallel_nd(p.N * p.ldc, [&](int i) { C_ref[i] = C[i]; });

    auto status = mkldnn_sgemm(&p.transA, &p.transB, &p.M, &p.N, &p.K, &p.alpha,
        A, &p.lda, B, &p.ldb, &p.beta, C, &p.ldc);
    if (status == mkldnn_success) {
        ref_gemm(&p.transA, &p.transB, p.M, p.N, p.K, p.alpha, A, p.lda, B, p.ldb,
            p.beta, C_ref, p.ldc);
        compare<float, float>(p.M, p.N, C, C_ref, p.ldc);
    }

    test_free((char *)A);
    test_free((char *)B);
    test_free((char *)C);
    test_free((char *)C_ref);

    if (status != mkldnn_success)
        throw error(status, "mkldnn_sgemm returned error");
}

template <typename a_dt, typename b_dt, typename c_dt>
class gemm_test_common: public ::testing::TestWithParam<test_params> {
protected:
    virtual void SetUp() {
        test_params p
            = ::testing::TestWithParam<test_params>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }
    virtual void Test() {
        test_params p = ::testing::TestWithParam<test_params>::GetParam();
        run_test_gemm<a_dt, b_dt, c_dt>(p);
    }
};
}
#endif
