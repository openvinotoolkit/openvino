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

#ifndef TEST_GEMM_DATA_PREPARATION_H
#define TEST_GEMM_DATA_PREPARATION_H

#include "test_gemm_params.hpp"

#include "dnnl_test_common.hpp"

namespace dnnl {
/*
 * To reduce the time spent in GEMM validation the test matrices A, B, and C
 * are generated from sub-matrices (A', B', and C') of smaller size:
 * - A(M, K) <-> A'(M_test, K)
 * - B(K, N) <-> B'(K, N_test)
 * - C(M, N) <-> C'(M_test, N_test)
 *
 * The matrices A', B', and C' are generated randomly. Then:
 * - A(m, k) := A'(mapper_m[m], k),
 * - B(k, n) := B'(k, mapper_n[n]),
 * - C(m, n) := C'(mapper_m[m], mapper_n[n]);
 *
 * Here `mapper_x[]` is surjection of {0, ..., X-1} onto {0, ..., X_test-1}.
 * For simplicity mapper_x[x] = x, for x in {0, ..., X_test-1}.
 *
 * This technique allows reducing the complexity of the validation code from
 * O(M*N*K) to O(M_test * N_test * K).
 *
 * X_test := min(X, X_test_max), where X_test_max is prime number around 50.
 *
 * To make the test robust the surjective functions mapper_m and mapper_n
 * should randomly map the elements {X_test, ..., X-1} onto {0, ..., X_test-1}.
 */

static constexpr int M_test_max = 47;
static constexpr int N_test_max = 53;

/** Mapper:
 * a surjective function from {0, ..., dim-1} onto {0, ..., dim_test-1}.
 */
struct mapper_t {
    mapper_t(const mapper_t &other)
        : dim_(other.dim_)
        , dim_test_(other.dim_test_)
        , gen_(other.gen_)
        , gen_start_(other.gen_start_)
        , mapper_(other.mapper_) {}

    mapper_t(mapper_t &&other) noexcept
        : dim_(other.dim_)
        , dim_test_(other.dim_test_)
        , gen_(other.gen_)
        , gen_start_(other.gen_start_)
        , mapper_(std::move(other.mapper_)) {}

    mapper_t(int64_t dim, int64_t dim_test_max, int64_t gen = 7,
            int64_t gen_start = 13)
        : dim_(dim)
        , dim_test_((std::min)(dim, dim_test_max))
        , gen_(gen)
        , gen_start_(gen_start)
        , mapper_(dim) {
        for (int64_t d = 0; d < dim_test_; ++d)
            mapper_[d] = d;
        for (int64_t g = gen_start_ % dim_test_, d = dim_test_; d < dim_; ++d) {
            mapper_[d] = mapper_[g];
            g = g * gen_ % dim_test_;
        }
    }

    int64_t dim() const { return dim_; }
    int64_t dim_test() const { return dim_test_; }
    int64_t operator[](int64_t d) const { return mapper_[d]; }

private:
    const int64_t dim_;
    const int64_t dim_test_;
    const int64_t gen_, gen_start_;
    std::vector<int64_t> mapper_;
};

struct test_gemm_data {
    std::shared_ptr<test_memory> a_mem;
    std::shared_ptr<test_memory> b_mem;
    std::shared_ptr<test_memory> c_mem;
    std::shared_ptr<test_memory> c_ref_mem;
    std::shared_ptr<test_memory> oc_mem;
    std::shared_ptr<mapper_t> mapper_m;
    std::shared_ptr<mapper_t> mapper_n;
};

/** Prepares matrix A or B according to the dimension mapper.
 * The K dimension is always assumed to be columns, hence:
 * - A layout = A_is_transposed ? ROW_MAJOR : COL_MAJOR
 * - B layout = B_is_transposed ? COL_MAJOR : ROW_MAJOR
 */
template <typename data_t>
void prepare_matrix(const test_memory &M_mem, int64_t off_beg, layout_t layout,
        int64_t R, int64_t C, int64_t LD, const mapper_t &mapper) {
    auto M = map_memory<data_t>(M_mem);
    auto dt = data_traits<data_t>::data_type;
    bool is_fp = (false || dt == memory::data_type::f16
            || dt == memory::data_type::bf16 || dt == memory::data_type::f32);
    const data_t mean = (data_t)(is_fp ? 1.f : 4);
    const data_t var = (data_t)(is_fp ? 2e-1f : 3);

    ASSERT_EQ(R, mapper.dim());
    const int R_test = mapper.dim_test();

    if (layout == layout_t::COL_MAJOR) {
        dnnl::impl::parallel_nd(C, R_test, [&](int64_t c, int64_t r) {
            const int64_t off = c * LD + r;
            M[off_beg + off] = set_value<data_t>(off, mean, var, 1.);
        });
        if (R > R_test) {
            const int64_t R_rest = R - R_test;
            dnnl::impl::parallel_nd(C, R_rest, [&](int64_t c, int64_t r_) {
                const int64_t r = R_test + r_;
                const int64_t off = c * LD + r;
                const int64_t off0 = c * LD + mapper[r];
                M[off_beg + off] = M[off_beg + off0];
            });
        }
    } else {
        dnnl::impl::parallel_nd(R_test, C, [&](int64_t r, int64_t c) {
            const int64_t off = r * LD + c;
            M[off_beg + off] = set_value<data_t>(off, mean, var, 1.);
        });
        if (R > R_test) {
            const int64_t R_rest = R - R_test;
            dnnl::impl::parallel_nd(R_rest, C, [&](int64_t r_, int64_t c) {
                const int64_t r = R_test + r_;
                const int64_t off = r * LD + c;
                const int64_t off0 = mapper[r] * LD + c;
                M[off_beg + off] = M[off_beg + off0];
            });
        }
    }

    // To test if igemm row/col sum are correct when performing sign/zero
    // extensions.
    if (dt == memory::data_type::u8)
        M[off_beg] = data_t(UINT8_MAX);
    else if (dt == memory::data_type::s8)
        M[off_beg] = data_t(-64);
}

/** Extends columns of the matrix M according to the mapper_c */
template <typename data_t>
void extend_matrix_cols(const test_memory &M_mem, int64_t off, int64_t R,
        int64_t C, int64_t LD, const mapper_t &mapper_c) {
    auto M = map_memory<data_t>(M_mem);
    ASSERT_EQ(C, mapper_c.dim());
    const int64_t C_test = mapper_c.dim_test();
    if (C_test == C) return;

    dnnl::impl::parallel_nd(R, C - C_test, [&](int64_t r, int64_t c_) {
        const int64_t c = C_test + c_;
        const int64_t c0 = mapper_c[c];
        M[off + r * LD + c] = M[off + r * LD + c0];
    });
}

/** Extends rows of the matrix M according to the mapper_r */
template <typename data_t>
void extend_matrix_rows(const test_memory &M_mem, int64_t off, int64_t R,
        int64_t C, int64_t LD, const mapper_t &mapper_r) {
    auto M = map_memory<data_t>(M_mem);
    ASSERT_EQ(R, mapper_r.dim());
    const int64_t R_test = mapper_r.dim_test();
    if (R_test == R) return;

    dnnl::impl::parallel_nd(R - R_test, [&](int64_t r_) {
        const int64_t r = R_test + r_;
        const int64_t r0 = mapper_r[r];
        for (int64_t c = 0; c < C; ++c)
            M[off + r * LD + c] = M[off + r0 * LD + c];
    });
}

/** Extends matrix M according to the mapper_r and mapper_c */
template <typename data_t>
void extend_matrix(const test_memory &M_mem, int64_t off, int64_t R, int64_t C,
        int64_t LD, const mapper_t &mapper_r, const mapper_t &mapper_c) {
    ASSERT_EQ(R, mapper_r.dim());
    ASSERT_EQ(C, mapper_c.dim());
    extend_matrix_rows<data_t>(M_mem, off, R, C, LD, mapper_r);
    extend_matrix_cols<data_t>(M_mem, off, R, C, LD, mapper_c);
}

inline void get_matrix_size(
        const test_params &p, size_t &sizeA, size_t &sizeB, size_t &sizeC) {
    const bool tr_a = (p.transA == 'T' || p.transA == 't');
    const bool tr_b = (p.transB == 'T' || p.transB == 't');
    sizeA = tr_a ? p.lda * p.K : p.lda * p.M;
    sizeB = tr_b ? p.ldb * p.N : p.ldb * p.K;
    sizeC = p.ldc * p.M;
}

template <typename T>
inline memory::desc get_matrix_md(memory::dim n, memory::dim off) {
    return create_md(
            {n + off}, data_traits<T>::data_type, memory::format_tag::x);
}

template <typename a_dt, typename b_dt, typename c_dt>
void fill_matrices(const test_params &p, const mapper_t &mapper_m,
        const mapper_t &mapper_n, const test_memory &a_mem,
        const test_memory &b_mem, const test_memory &c_mem,
        const test_memory &c_ref_mem, const test_memory &oc_mem) {
    prepare_matrix<a_dt>(a_mem, p.off.a,
            p.tr_a() ? layout_t::COL_MAJOR : layout_t::ROW_MAJOR, p.M, p.K,
            p.lda, mapper_m);
    prepare_matrix<b_dt>(b_mem, p.off.b,
            p.tr_b() ? layout_t::ROW_MAJOR : layout_t::COL_MAJOR, p.N, p.K,
            p.ldb, mapper_n);

    fill_data<c_dt>(p.off.c + p.sizeC(), c_mem.get());
    extend_matrix<c_dt>(c_mem, p.off.c, p.M, p.N, p.ldc, mapper_m, mapper_n);
    {
        auto C = map_memory<c_dt>(c_mem);
        auto C_ref = map_memory<c_dt>(c_ref_mem);
        dnnl::impl::parallel_nd(p.sizeC(),
                [&](int64_t i) { C_ref[p.off.c + i] = C[p.off.c + i]; });
    }

    if (oc_mem.get_size() == 0) return;

    if (p.igemm_params.nonzero_oc) {
        fill_data<c_dt>(p.size_oc(), oc_mem.get(), (c_dt)1, (c_dt)0);
        if (p.oc_is_R()) {
            extend_matrix_cols<c_dt>(oc_mem, 0, 1, p.N, p.N, mapper_n);
        } else if (p.oc_is_C()) {
            extend_matrix_rows<c_dt>(oc_mem, 0, p.M, 1, 1, mapper_m);
        }
    } else {
        auto oc = map_memory<c_dt>(oc_mem);
        for (int64_t i = 0; i < p.size_oc(); i++)
            oc[i] = 0;
    }
}

template <typename a_dt, typename b_dt, typename c_dt>
void prepare_data_for_gemm_testing(
        const test_params &p, test_gemm_data &gemm_data) {
    size_t sizeA, sizeB, sizeC;
    get_matrix_size(p, sizeA, sizeB, sizeC);

    engine eng = get_test_engine();
    gemm_data.a_mem.reset(
            new test_memory(get_matrix_md<a_dt>(sizeA, p.off.a), eng));
    gemm_data.b_mem.reset(
            new test_memory(get_matrix_md<b_dt>(sizeB, p.off.b), eng));
    gemm_data.c_mem.reset(
            new test_memory(get_matrix_md<c_dt>(sizeC, p.off.c), eng));
    gemm_data.c_ref_mem.reset(
            new test_memory(get_matrix_md<c_dt>(sizeC, p.off.c), eng));
    gemm_data.oc_mem.reset(
            new test_memory(get_matrix_md<c_dt>(p.size_oc(), p.off.co), eng));

    gemm_data.mapper_m.reset(new mapper_t(p.M, M_test_max));
    gemm_data.mapper_n.reset(new mapper_t(p.N, N_test_max));

    fill_matrices<a_dt, b_dt, c_dt>(p, *gemm_data.mapper_m, *gemm_data.mapper_n,
            *gemm_data.a_mem, *gemm_data.b_mem, *gemm_data.c_mem,
            *gemm_data.c_ref_mem, *gemm_data.oc_mem);
}

} // namespace dnnl

#endif
