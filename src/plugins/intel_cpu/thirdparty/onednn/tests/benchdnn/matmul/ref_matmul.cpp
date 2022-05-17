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

#include "tests/test_thread.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

void compute_ref_matmul(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &bia_m = args.find(DNNL_ARG_BIAS);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;
    const int64_t MB = dst_m.nelems() / (M * N);
    const int batch_ndims = dst_m.md_.ndims - 2;

    const int wei_zero_point = prb->attr.zero_points[DNNL_ARG_WEIGHTS];

    dnn_mem_t dst_tmp(dst_m, dnnl_f32, tag::undef, dst_m.engine());

    const auto src_broadcast_mask = prb->src_broadcast_mask();
    const auto wei_broadcast_mask = prb->weights_broadcast_mask();

    dnnl::impl::parallel_nd(MB, M, N, [&](int64_t mb, int64_t m, int64_t n) {
        auto src = (const float *)src_m;
        auto wei = (const float *)wei_m;

        float dst = 0;
        const int64_t src_mb
                = dst_m.get_scale_idx(mb, src_broadcast_mask, batch_ndims);
        const int64_t wei_mb
                = dst_m.get_scale_idx(mb, wei_broadcast_mask, batch_ndims);
        for (int64_t k = 0; k < K; ++k) {
            auto s = src[src_off_f(prb, src_mb, m, k)];
            maybe_zero_point(prb->attr, s, prb->src_zp, k, DNNL_ARG_SRC);
            dst += s * (wei[wei_off_f(prb, wei_mb, k, n)] - wei_zero_point);
        }
        ((float *)dst_tmp)[dst_off_f(prb, mb, m, n)] = dst;
    });

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    const auto bias_broadcast_mask = prb->bias_broadcast_mask();
    dnnl::impl::parallel_nd(MB, M, N, [&](int64_t mb, int64_t m, int64_t n) {
        size_t dst_off = dst_off_f(prb, mb, m, n);
        float &dst = ((float *)dst_m)[dst_off];

        float tmp = ((float *)dst_tmp)[dst_off];
        if (prb->bia_dt != dnnl_data_type_undef) {
            int64_t bia_off = dst_m.get_scale_idx(dst_off, bias_broadcast_mask);
            float *bia_ptr = (float *)bia_m;
            tmp += bia_ptr[bia_off];
        }
        maybe_oscale(prb->attr, tmp, prb->scales, n);

        const auto v_po_vals
                = prepare_po_vals(dst_m, args, v_po_masks, dst_off);

        maybe_post_ops(prb->attr, tmp, dst, v_po_vals);

        maybe_zero_point(prb->attr, tmp, prb->dst_zp, n, DNNL_ARG_DST, true);
        dst = tmp;
    });
}

void compute_ref(
        const prb_t *prb, dnnl_primitive_t prim_ref, const args_t &args) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    compute_ref_matmul(prb, args);
}

} // namespace matmul
