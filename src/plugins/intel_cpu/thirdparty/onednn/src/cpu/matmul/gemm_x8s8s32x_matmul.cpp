/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <atomic>

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/matmul/gemm_x8s8s32x_matmul.hpp"
#include "cpu/matmul/matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

using namespace data_type;

namespace {
template <typename pd_t>
bool need_post_processing(const pd_t *pd, float runtime_dst_zero_point = 0.f) {
    return pd->with_bias() || pd->dst_md()->data_type != s32
            || !pd->params().dst_is_acc_
            || !pd->params().pp_attr_.has_default_values()
            || !pd->params().pp_attr_.zero_points_.has_default_values(
                    DNNL_ARG_DST)
            || runtime_dst_zero_point != 0.f;
}
} // namespace

status_t gemm_x8s8s32x_matmul_t::pd_t::init(engine_t *engine) {
    using namespace utils;
    using namespace data_type;

    auto check_attr_oscale = [&]() -> bool {
        const auto &oscale = attr()->output_scales_;
        return oscale.mask_ == 0
                || (oscale.mask_ == (1 << (dst_md()->ndims - 1)));
    };

    auto check_attr_zero_points
            = [&]() -> bool { return attr()->zero_points_.common(); };

    auto check_attr_post_ops = [&]() -> bool {
        using namespace primitive_kind;
        const auto &post_ops = attr()->post_ops_;
        static const bcast_set_t enabled_bcast_strategy {
                broadcasting_strategy_t::scalar,
                broadcasting_strategy_t::per_oc,
                broadcasting_strategy_t::per_oc_spatial,
                broadcasting_strategy_t::per_mb_spatial,
                broadcasting_strategy_t::no_broadcast};
        const bool is_binary_po_per_oc
                = binary_injector_utils::bcast_strategy_present(
                        binary_injector_utils::extract_bcast_strategies(
                                post_ops.entry_, dst_md()),
                        broadcasting_strategy_t::per_oc);
        return cpu::inner_product_utils::post_ops_ok(
                       post_ops, dst_md(), enabled_bcast_strategy)
                && IMPLICATION(is_binary_po_per_oc,
                        gemm_based::check_gemm_binary_per_oc_compatible_formats(
                                *this));
    };

    bool ok = one_of(src_md()->data_type, s8, u8)
            && weights_md()->data_type == s8 && desc()->accum_data_type == s32
            && one_of(dst_md()->data_type, f32, s32, s8, u8)
            && IMPLICATION(with_bias(),
                    one_of(weights_md(1)->data_type, f32, s32, s8, u8)
                            && is_bias_1xN())
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale_runtime
                            | primitive_attr_t::skip_mask_t::zero_points_runtime
                            | primitive_attr_t::skip_mask_t::post_ops
                            | primitive_attr_t::skip_mask_t::sum_dt,
                    dst_md()->data_type)
            && attr_.post_ops_.check_sum_consistent_dt(dst_md()->data_type)
            // need to set up default formats first, so that latter checks can
            // be perfomed properly
            && set_default_formats() && check_attr_oscale()
            && check_attr_zero_points() && check_attr_post_ops()
            && gemm_based::check_gemm_compatible_formats(*this)
            && attr_.set_default_formats(dst_md(0)) == status::success;
    if (!ok) return status::unimplemented;

    // set states

    // copy attributes and drop src and weights zero points
    CHECK(params_.pp_attr_.copy_from(*attr()));
    params_.pp_attr_.zero_points_.set(DNNL_ARG_SRC, 0);
    params_.pp_attr_.zero_points_.set(DNNL_ARG_WEIGHTS, 0);

    params_.gemm_applies_output_scales_ = false;
    params_.gemm_beta_ = 0.f;

    bool do_sum = params_.pp_attr_.post_ops_.find(primitive_kind::sum) >= 0;
    params_.dst_is_acc_
            = utils::one_of(dst_md()->data_type, s32, f32) && !do_sum;

    params_.has_pp_kernel_ = need_post_processing(this);

    gemm_based::book_acc_scratchpad(*this, params_, sizeof(int32_t));

    return status::success;
}

void gemm_x8s8s32x_matmul_t::post_process_src_and_weights_zero_points(
        std::vector<int32_t> &src_comp, std::vector<int32_t> &wei_comp, dim_t M,
        dim_t N, dim_t K, const char *src, dim_t src_s0, dim_t src_s1,
        const int8_t *wei, dim_t wei_s0, dim_t wei_s1, int32_t *acc, int ldc,
        int32_t src_zero_point, int32_t wei_zero_point) const {
    if (wei_zero_point) {
        for_(dim_t m = 0; m < M; ++m)
        for (dim_t k = 0; k < K; ++k) {
            if (k == 0) src_comp[m] = int32_t(0);
            src_comp[m] += src[src_s0 * m + src_s1 * k];
        }
    }

    if (src_zero_point) {
        for_(dim_t k = 0; k < K; ++k)
        for (dim_t n = 0; n < N; ++n) {
            if (k == 0) wei_comp[n] = int32_t(0);
            wei_comp[n] += wei[wei_s0 * k + wei_s1 * n];
        }
    }

    for_(dim_t m = 0; m < M; ++m)
    for (dim_t n = 0; n < N; ++n)
        acc[m * ldc + n] += 0 - src_zero_point * wei_comp[n]
                - wei_zero_point * src_comp[m]
                + src_zero_point * wei_zero_point * (int)K;
}

status_t gemm_x8s8s32x_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    using namespace binary_injector_utils;

    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    const auto &po = this->pd()->attr()->post_ops_;
    const auto post_ops_binary_rhs_arg_vec = prepare_binary_args(po, ctx);

    DEFINE_SCALES_BUFFER(scales);
    DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(weights_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, DNNL_ARG_DST);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    char gemm_off_a = static_cast<char>(src_zero_point);
    int8_t gemm_off_b = static_cast<int8_t>(weights_zero_point);
    bool post_process_src_and_weights_zero_points_outside_of_gemm = false;
    if (gemm_off_a != src_zero_point || gemm_off_b != weights_zero_point) {
        post_process_src_and_weights_zero_points_outside_of_gemm = true;
        gemm_off_a = gemm_off_b = 0;
    }
    const float dst_zero_point_f32 = static_cast<float>(dst_zero_point);

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const int batch_ndims = ndims - 2;
    dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();
    const dim_t batch_without_dim0
            = helper.ndims() > 3 ? batch / dst_d.dims()[0] : 0;
    const dim_t batch_without_dim01
            = helper.ndims() > 4 ? batch_without_dim0 / dst_d.dims()[1] : 1;
    const char transA = helper.transA();
    const char transB = helper.transB();
    const dim_t lda = helper.lda();
    const dim_t ldb = helper.ldb();
    const dim_t ldc = helper.ldc();
    const int ldx_dim_idx = pd()->ndims() - 2;
    const dim_t *src_strides = &src_d.blocking_desc().strides[ldx_dim_idx];
    const dim_t *weights_strides
            = &weights_d.blocking_desc().strides[ldx_dim_idx];

    const gemm_based::params_t &params = pd()->params();
    const bool can_fuse_src_batch_dims = pd()->has_runtime_dims_or_strides()
            ? helper.can_fuse_src_batch_dims()
            : params.can_fuse_src_batch_dims_;
    const dim_t acc_stride = gemm_based::get_scratchpad_size(
            batch, M, N, can_fuse_src_batch_dims);
    bool dst_is_acc = params.dst_is_acc_;
    int32_t *acc = dst_is_acc
            ? reinterpret_cast<int32_t *>(dst)
            : ctx.get_scratchpad_grantor().template get<int32_t>(
                    memory_tracking::names::key_matmul_dst_in_acc_dt);
    // case: dynamic sizes
    bool need_free_acc = false;
    if (acc == nullptr) {
        acc = (int32_t *)malloc(sizeof(int32_t) * acc_stride
                        * ((can_fuse_src_batch_dims || batch == 1)
                                        ? 1
                                        : (dim_t)dnnl_get_max_threads()),
                64);

        if (acc == nullptr) return status::out_of_memory;
        need_free_acc = true;
    }

    const float alpha = params.get_gemm_alpha(scales);
    const float beta = params.gemm_beta_;
    const dim_t acc_ldc = dst_is_acc ? ldc : N;
    const int scale_idx_mult
            = this->pd()->attr()->output_scales_.mask_ == (1 << (ndims - 1));

    std::atomic<status_t> st(status::success);
    // use parallel over batch when binary po with channel bcast
    // (except batch == 1)
    bool is_binary_po_per_oc;
    bool is_binary_po_per_oc_sp;
    bool is_binary_po_channel_bcast;
    std::tie(is_binary_po_per_oc, is_binary_po_per_oc_sp,
            is_binary_po_channel_bcast)
            = bcast_strategies_present_tup(po.entry_, pd()->dst_md(),
                    broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::per_oc_spatial,
                    broadcasting_strategy_t::per_mb_spatial);
    // if batched, parralel over batch for per_mb_sp and per_oc binary
    // post-op broadcast
    const bool can_use_po_with_fused_batch = !is_binary_po_channel_bcast
            && IMPLICATION(
                    is_binary_po_per_oc || is_binary_po_per_oc_sp, ndims == 2);
    const bool parallel_over_batch = batch > 1 && !can_fuse_src_batch_dims;
    if (IMPLICATION(can_use_po_with_fused_batch, parallel_over_batch)) {
        const int src_mask
                = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
        const int wei_mask
                = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);
        const size_t bia_dt_size = !pd()->with_bias()
                ? 0
                : types::data_type_size(pd()->weights_md(1)->data_type);
        const size_t dst_dt_size = types::data_type_size(dst_d.data_type());
        const size_t work_amount = (size_t)batch * M * N;
        const size_t work_per_batch = (size_t)M * N;

        // NOTE: inside lambda, type cast variables captured by reference using
        // either c-like "(type)var" or functional "type(var)" notation in order
        // to avoid gcc bug with c++14 standard. Otherwise, capture by value.
        parallel(0, [=, &st](int ithr, int nthr) {
            size_t t_work_start {0}, t_work_end {0};
            balance211(work_amount, nthr, ithr, t_work_start, t_work_end);

            dim_t cur_b {0}, cur_m {0}, cur_n {0};
            dims_t s_dims_idx, w_dims_idx, d_dims_idx;
            size_t i_work = t_work_start;

            const bool reuse_acc = acc != (int32_t *)dst;
            int32_t *curr_acc = reuse_acc ? acc + ithr * acc_stride : nullptr;

            std::vector<int32_t> src_compensation(M, 0);
            std::vector<int32_t> weights_compensation(N, 0);

            // icc 17.0 has a bug with capturing const variables with value known
            // at compilation time in lambdas
            const int32_t gemm_off_c = 0;

            while (i_work < t_work_end) {
                utils::nd_iterator_init(
                        i_work, cur_b, batch, cur_m, M, cur_n, N);

                utils::l_dims_by_l_offset(
                        d_dims_idx, i_work, dst_d.dims(), ndims);

                utils::copy_dims_with_mask(
                        s_dims_idx, d_dims_idx, batch_ndims, src_mask);
                s_dims_idx[ndims - 2] = cur_m;
                s_dims_idx[ndims - 1] = 0; // k idx is always 0

                utils::copy_dims_with_mask(
                        w_dims_idx, d_dims_idx, batch_ndims, wei_mask);
                w_dims_idx[ndims - 2] = 0; // k idx is always 0
                w_dims_idx[ndims - 1] = cur_n;

                const char *curr_src = src + src_d.off_v(s_dims_idx);
                const int8_t *curr_weights
                        = weights + weights_d.off_v(w_dims_idx);
                const dim_t dst_off = dst_d.off_v(d_dims_idx);
                char *curr_dst = dst + dst_dt_size * dst_off;
                if (!reuse_acc) curr_acc = acc + dst_off;

                dim_t gemm_M {0}, gemm_N {0};
                size_t matrix_offset;
                const size_t rem_work = t_work_end - i_work;
                if (rem_work >= work_per_batch && cur_m == 0 && cur_n == 0) {
                    // parallel over batch
                    gemm_M = M;
                    gemm_N = N;
                    matrix_offset = 0;
                } else if (rem_work >= (size_t)N && cur_n == 0) {
                    // parallel over M
                    gemm_M = nstl::min(
                            (size_t)(M - cur_m), (size_t)(rem_work / N));
                    gemm_N = N;
                    matrix_offset = cur_n + cur_m * N;
                } else {
                    // parallel over N
                    gemm_M = 1;
                    gemm_N = nstl::min((size_t)(N - cur_n), rem_work);
                    matrix_offset = cur_n + cur_m * N;
                }

                status_t st_thr = status::runtime_error;
                switch (src_d.data_type()) {
                    case data_type::s8: {
                        const int8_t *curr_src_
                                = reinterpret_cast<const int8_t *>(curr_src);
                        int8_t gemm_off_a_ = static_cast<int8_t>(gemm_off_a);
                        st_thr = gemm_s8x8s32(&transB, &transA, "F", &gemm_N,
                                &gemm_M, &K, &alpha, curr_weights, &ldb,
                                &gemm_off_b, curr_src_, &lda, &gemm_off_a_,
                                &beta, curr_acc, &acc_ldc, &gemm_off_c);
                    } break;
                    case data_type::u8: {
                        const uint8_t *curr_src_
                                = reinterpret_cast<const uint8_t *>(curr_src);
                        uint8_t gemm_off_a_ = static_cast<uint8_t>(gemm_off_a);
                        st_thr = gemm_s8x8s32(&transB, &transA, "F", &gemm_N,
                                &gemm_M, &K, &alpha, curr_weights, &ldb,
                                &gemm_off_b, curr_src_, &lda, &gemm_off_a_,
                                &beta, curr_acc, &acc_ldc, &gemm_off_c);
                    } break;
                    default: assert(!"unsupported data type"); break;
                }

                if (st_thr != status::success) {
                    st = st_thr;
                    return;
                }

                // if igemm cannot handle src and weights zero points
                if (post_process_src_and_weights_zero_points_outside_of_gemm) {
                    post_process_src_and_weights_zero_points(src_compensation,
                            weights_compensation, gemm_M, gemm_N, K, curr_src,
                            src_strides[0], src_strides[1], curr_weights,
                            weights_strides[0], weights_strides[1], curr_acc,
                            acc_ldc, src_zero_point, weights_zero_point);
                }

                bool postops_in_matmul
                        = need_post_processing(pd(), dst_zero_point_f32);
                assert(IMPLICATION(postops_in_matmul, params.has_pp_kernel_));

                if (postops_in_matmul) {
                    const size_t dst_logical_off = i_work;
                    const size_t dim1_off = helper.ndims() > 3
                            ? ((cur_b % batch_without_dim0)
                                    / batch_without_dim01)
                            : cur_m;
                    // offset for case with post-op broadcast_channel
                    const size_t matrix_per_first_batch_off = helper.ndims() > 3
                            ? M * N * (cur_b / batch_without_dim0)
                                    + matrix_offset
                            : 0;
                    const ptrdiff_t oc_off = i_work % N;
                    (*pp_kernel_)(curr_dst, curr_acc,
                            bias + oc_off * bia_dt_size,
                            scales + oc_off * scale_idx_mult, 0,
                            dst_logical_off, dim1_off, gemm_M * gemm_N,
                            static_cast<size_t>(N), ldc, &dst_zero_point_f32,
                            post_ops_binary_rhs_arg_vec.data(), dst,
                            matrix_per_first_batch_off, ctx, *pd()->dst_md());
                }
                i_work += gemm_M * gemm_N;
            }
        });
    } else {
        // icc 17.0 has a bug with capturing const variables with value known
        // at compilation time in lambdas
        const int32_t gemm_off_c = 0;

        // collapse batch into M, if weights batch dimensions are broadcasted.
        M = batch * M;
        status_t st = status::runtime_error;
        switch (src_d.data_type()) {
            case data_type::s8: {
                const int8_t *src_ = reinterpret_cast<const int8_t *>(src);
                int8_t gemm_off_a_ = static_cast<int8_t>(gemm_off_a);
                st = gemm_s8x8s32(&transB, &transA, "F", &N, &M, &K, &alpha,
                        weights, &ldb, &gemm_off_b, src_, &lda, &gemm_off_a_,
                        &beta, acc, &acc_ldc, &gemm_off_c);
            } break;
            case data_type::u8: {
                const uint8_t *src_ = reinterpret_cast<const uint8_t *>(src);
                uint8_t gemm_off_a_ = static_cast<uint8_t>(gemm_off_a);
                st = gemm_s8x8s32(&transB, &transA, "F", &N, &M, &K, &alpha,
                        weights, &ldb, &gemm_off_b, src_, &lda, &gemm_off_a_,
                        &beta, acc, &acc_ldc, &gemm_off_c);
            } break;
            default: assert(!"unsupported data type"); break;
        }

        if (st == status::success) {
            std::vector<int32_t> src_compensation(M, 0);
            std::vector<int32_t> weights_compensation(N, 0);

            // if igemm cannot handle src and weights zero points
            if (post_process_src_and_weights_zero_points_outside_of_gemm) {
                post_process_src_and_weights_zero_points(src_compensation,
                        weights_compensation, M, N, K, src, src_strides[0],
                        src_strides[1], weights, weights_strides[0],
                        weights_strides[1], acc, acc_ldc, src_zero_point,
                        weights_zero_point);
            }

            bool postops_in_matmul
                    = need_post_processing(pd(), dst_zero_point_f32);
            assert(IMPLICATION(postops_in_matmul, params.has_pp_kernel_));

            if (postops_in_matmul) {
                const bool force_sequential = pp_kernel_->sequential_kernel();
                parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
                    size_t start {}, end {};
                    balance211((size_t)(M * N), nthr, ithr, start, end);
                    const size_t dst_logical_off = start;
                    const size_t dim1_off = start % N;
                    (*pp_kernel_)(dst, acc, bias, scales, start,
                            dst_logical_off, dim1_off, end, (size_t)N, ldc,
                            &dst_zero_point_f32,
                            post_ops_binary_rhs_arg_vec.data(), dst, 0, ctx,
                            *pd()->dst_md());
                });
            }
        }
    }
    if (need_free_acc) free(acc);

    return st;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
