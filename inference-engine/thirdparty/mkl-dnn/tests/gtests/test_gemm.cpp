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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn_types.h"
#include "mkldnn.h"

namespace mkldnn {
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

    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

void ref_gemm(const char *transa, const char *transb, int m, int n, int k,
        const float alpha, const float *a, int lda, const float *b,
        int ldb, float beta, float *c, int ldc) {

    const bool tr_a = transa && (*transa == 'T' || *transa == 't');
    const bool tr_b = transb && (*transb == 'T' || *transb == 't');

    auto pa = [=] (int i, int j) { return a[j*lda + i]; };
    auto pb = [=] (int i, int j) { return b[j*ldb + i]; };
    auto pc = [=] (int i, int j) { return c[j*ldc + i]; };

#   pragma omp parallel for collapse(2)
    for (int im = 0; im < m; im++) {
        for (int in = 0; in < n; in++) {
            float c_elem = (beta == 0.) ? 0. : pc(im, in) * beta;
            for (int ik = 0; ik < k; ik++) {
                const float a_elem = tr_a ? pa(ik, im) : pa(im, ik);
                const float b_elem = tr_b ? pb(in, ik) : pb(ik, in);
                c_elem += alpha * a_elem * b_elem;
            }
            c[in*ldc + im] = c_elem;
        }
    }
}

void compare(int M, int N, int ldc, float *C, float *C_ref) {
#   pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < ldc; ++j) {
            float ref = C_ref[i*ldc + j];
            float got = C[i*ldc + j];
            float diff = got - ref;
            float e = (std::abs(ref) > 1e-4) ? diff / ref : diff;
            EXPECT_NEAR(e, 0.0, 1e-4)
                << "Row: " << j << " Column: " << i;
        }
    }
}

class sgemm_test: public ::testing::TestWithParam<test_params> {
protected:
    virtual void SetUp() {
        test_params p
            = ::testing::TestWithParam<test_params>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }
    virtual void Test() {
        mkldnn_status_t status;
        test_params p
            = ::testing::TestWithParam<test_params>::GetParam();
        const bool tr_a = (p.transA == 'T' || p.transA == 't');
        const bool tr_b = (p.transB == 'T' || p.transB == 't');
        size_t sizeA = !tr_a ? p.lda * p.K : p.lda * p.M,
                sizeB = !tr_b ? p.ldb * p.N : p.ldb * p.K,
                sizeC = p.ldc * p.N;
        float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
        A = (float *)test_malloc(sizeA*sizeof(float));
        B = (float *)test_malloc(sizeB*sizeof(float));
        C = (float *)test_malloc(sizeC*sizeof(float));
        C_ref = (float *)test_malloc(sizeC*sizeof(float));

        fill_data<float>(sizeA, A);
        fill_data<float>(sizeB, B);
        fill_data<float>(sizeC, C);

        #pragma omp parallel for
        for (int i=0; i<p.N*p.ldc; i++)
                C_ref[i] = C[i];

        status = mkldnn_sgemm(&p.transA, &p.transB, &p.M, &p.N, &p.K, &p.alpha, A,
                &p.lda, B, &p.ldb, &p.beta, C, &p.ldc);
        if (status != mkldnn_success)
            throw error(status, "mkldnn_sgemm returned error");

        ref_gemm(&p.transA, &p.transB, p.M, p.N, p.K, p.alpha, A, p.lda,
                B, p.ldb, p.beta, C_ref, p.ldc);
        compare(p.M, p.N, p.ldc, C, C_ref);

        test_free((char *)A);
        test_free((char *)B);
        test_free((char *)C);
        test_free((char *)C_ref);
    }
};
TEST_P(sgemm_test, TestSGEMM) {}
INSTANTIATE_TEST_CASE_P(TestSGEMM, sgemm_test, ::testing::Values(
    test_params{'n', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, true, mkldnn_invalid_arguments},
    test_params{'t', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, true, mkldnn_invalid_arguments},
    test_params{'n', 't', 3, 2, 1, 1.0, 0.0, 3, 1, 8, true, mkldnn_invalid_arguments},
    test_params{'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, true, mkldnn_invalid_arguments},

    test_params{'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'t', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, false},
    test_params{'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'t', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'t', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'n', 'n', 2, 2, 10000, 1.0, 2.0, 2, 10000, 2, false},

    test_params{'n', 'n', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{'n', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},
    test_params{'t', 'n', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{'t', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},
    test_params{'n', 't', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{'n', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},
    test_params{'t', 't', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{'t', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false}
));
}
