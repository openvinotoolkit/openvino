/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef TEST_GEMM_DATA_VALIDATION_H
#define TEST_GEMM_DATA_VALIDATION_H

#include "test_gemm_params.hpp"

#include "dnnl_test_common.hpp"

namespace dnnl {

template <typename a_dt, typename b_dt, typename c_dt>
struct ref_gemm {
    static void call(const test_params &p, int64_t M, int64_t N,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem, const test_memory &) {
        auto a = map_memory<a_dt>(a_mem);
        auto b = map_memory<b_dt>(b_mem);
        auto c = map_memory<c_dt>(c_mem);

        const bool tr_a = p.transA && (p.transA == 'T' || p.transA == 't');
        const bool tr_b = p.transB && (p.transB == 'T' || p.transB == 't');

        auto pa = [&](int64_t i, int64_t j) {
            return a[p.off.a + i * p.lda + j];
        };
        auto pb = [&](int64_t i, int64_t j) {
            return b[p.off.b + i * p.ldb + j];
        };
        auto pc = [&](int64_t i, int64_t j) -> c_dt & {
            return c[p.off.c + i * p.ldc + j];
        };

        dnnl::impl::parallel_nd(M, N, [&](int64_t im, int64_t in) {
            c_dt c_elem = (p.beta == 0.) ? 0. : pc(im, in) * p.beta;

            for (int64_t ik = 0; ik < p.K; ik++) {
                const a_dt a_elem = tr_a ? pa(ik, im) : pa(im, ik);
                const b_dt b_elem = tr_b ? pb(in, ik) : pb(ik, in);
                c_elem += p.alpha * a_elem * b_elem;
            }
            pc(im, in) = c_elem;
        });
    }
};

template <typename a_dt, typename b_dt>
struct ref_gemm<a_dt, b_dt, int32_t> {
    static void call(const test_params &p, int64_t M, int64_t N,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem, const test_memory &oc_mem) {
        auto A = map_memory<a_dt>(a_mem);
        auto B = map_memory<b_dt>(b_mem);
        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);

        const bool tr_a = p.transA && (p.transA == 'T' || p.transA == 't');
        const bool tr_b = p.transB && (p.transB == 'T' || p.transB == 't');
        bool OCisR = (p.igemm_params.offsetc == 'R'
                || p.igemm_params.offsetc == 'r');
        bool OCisC = (p.igemm_params.offsetc == 'C'
                || p.igemm_params.offsetc == 'c');

        auto pa = [&](int64_t i, int64_t j) {
            return (double)A[p.off.a + i * p.lda + j];
        };
        auto pb = [&](int64_t i, int64_t j) {
            return (double)B[p.off.b + i * p.ldb + j];
        };
        auto pc = [&](int64_t i, int64_t j) -> int32_t & {
            return C[p.off.c + i * p.ldc + j];
        };

        int8_t oa = p.igemm_params.oa();
        int8_t ob = p.igemm_params.ob();

        dnnl::impl::parallel_nd(M, N, [&](int64_t m, int64_t n) {
            double c_elem = 0;
            for (int64_t k = 0; k < p.K; k++) {
                const double a_elem = (tr_a ? pa(k, m) : pa(m, k)) - oa;
                const double b_elem = (tr_b ? pb(n, k) : pb(k, n)) - ob;
                c_elem += a_elem * b_elem;
            }

            double coffset = OCisR ? oc[n] : OCisC ? oc[m] : oc[0];
            double val = (p.beta == 0.f ? 0. : p.beta * (double)pc(m, n))
                    + p.alpha * c_elem + coffset;
            pc(m, n) = static_cast<int32_t>(
                    nearbyint(saturate<int32_t, double>(val)));
        });
    }
};

template <typename a_dt, typename c_dt>
void compare(const test_params &p, const test_memory &c_mem,
        const test_memory &c_ref_mem) {
    using data_type = memory::data_type;
    auto c = map_memory<c_dt>(c_mem);
    auto c_ref = map_memory<c_dt>(c_ref_mem);
    dnnl::impl::parallel_nd(p.M, p.ldc, [&](int64_t i, int64_t j) {
        if (is_current_test_failed()) return;

        c_dt ref = c_ref[p.off.c + i * p.ldc + j];
        c_dt got = c[p.off.c + i * p.ldc + j];
        c_dt diff = got - ref;

        if (data_traits<a_dt>::data_type == data_type::f16) {
            const float eps = 1e-3 * p.K;
            float e = (std::abs(ref) > eps) ? diff / ref : float(diff);
            ASSERT_NEAR(e, 0.0, eps) << "Row: " << i << " Col: " << j;
        } else if (data_traits<a_dt>::data_type == data_type::bf16) {
            const float eps = 1e-2 * p.K;
            float e = (std::abs(ref) > eps) ? diff / ref : float(diff);
            ASSERT_NEAR(e, 0.0, eps) << "Row: " << i << " Col: " << j;
        } else if (data_traits<a_dt>::data_type == data_type::f32) {
            c_dt e = (std::abs(ref) > 1e-4) ? c_dt(diff / ref) : diff;
            ASSERT_NEAR(e, 0.0, 1e-4) << "Row: " << i << " Col: " << j;
        } else {
            // igemm
            c_dt eps = 0;
            if (p.alpha == 1.0f) {
                eps = 1;
            } else if (data_traits<a_dt>::data_type == data_type::u8) {
                eps = p.K / 700 + 1;
            } else if (data_traits<a_dt>::data_type == data_type::s8) {
                eps = p.K / 350 + 1;
            }
            ASSERT_NEAR(diff, 0, eps) << "Row: " << i << " Col: " << j;
        }
    });
}

template <typename a_dt, typename b_dt, typename c_dt>
void validate(const test_params &p, test_gemm_data &gemm_data) {
    const int64_t M_test = gemm_data.mapper_m->dim_test();
    const int64_t N_test = gemm_data.mapper_n->dim_test();

    ref_gemm<a_dt, b_dt, c_dt>::call(p, M_test, N_test, *gemm_data.a_mem,
            *gemm_data.b_mem, *gemm_data.c_ref_mem, *gemm_data.oc_mem);
    extend_matrix<c_dt>(*gemm_data.c_ref_mem, p.off.c, p.M, p.N, p.ldc,
            *gemm_data.mapper_m, *gemm_data.mapper_n);
    compare<a_dt, c_dt>(p, *gemm_data.c_mem, *gemm_data.c_ref_mem);
}

} // namespace dnnl

#endif
