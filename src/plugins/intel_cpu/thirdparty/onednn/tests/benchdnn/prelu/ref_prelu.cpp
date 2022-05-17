/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <assert.h>

#include <algorithm>
#include "prelu/prelu.hpp"
#include "tests/test_thread.hpp"

namespace prelu {

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &dst) {
    const float *src_ptr = (const float *)src;
    const float *wei_ptr = (const float *)weights;
    float *dst_ptr = (float *)dst;

    const auto nelems = src.nelems();
    const auto broadcast_mask = prb->get_broadcast_mask();

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const auto wei_idx = src.get_scale_idx(i, broadcast_mask);
        float res = src_ptr[i] * (src_ptr[i] > 0 ? 1.f : wei_ptr[wei_idx]);
        maybe_saturate(prb->sdt[0], res);
        dst_ptr[i] = res;
    });
}

void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_weights) {
    const float *src_ptr = (const float *)src;
    const float *wei_ptr = (const float *)weights;
    const float *diff_dst_ptr = (const float *)diff_dst;
    float *diff_src_ptr = (float *)diff_src;
    float *diff_wei_ptr = (float *)diff_weights;

    const auto src_nelems = diff_src.nelems();
    const auto wei_nelems = diff_weights.nelems();
    float *diff_wei_buf = diff_wei_ptr;

    const auto ker = [&](int64_t i, int64_t wei_idx, int64_t d_wei_idx) {
        float d_src
                = diff_dst_ptr[i] * (src_ptr[i] > 0 ? 1.f : wei_ptr[wei_idx]);
        maybe_saturate(prb->sdt[0], d_src);
        diff_src_ptr[i] = d_src;
        diff_wei_buf[d_wei_idx] += MIN2(0.f, src_ptr[i]) * diff_dst_ptr[i];
    };

    dnnl::impl::parallel_nd(
            wei_nelems, [&](int64_t i) { diff_wei_ptr[i] = 0; });

    if (wei_nelems == 1) {
        const auto num_thr = dnnl_get_max_threads();
        diff_wei_buf = new float[num_thr];
        dnnl::impl::parallel(0, [&](const int ithr, const int nthr) {
            int64_t start {0}, end {0};
            dnnl::impl::balance211(src_nelems, nthr, ithr, start, end);
            diff_wei_buf[ithr] = 0;

            for (int64_t i = start; i < end; ++i)
                ker(i, 0, ithr);
        });

        for (int64_t i = 0; i < num_thr; i++)
            diff_wei_ptr[0] += diff_wei_buf[i];
        delete[] diff_wei_buf;

    } else if (src_nelems == wei_nelems) {
        dnnl::impl::parallel_nd(src_nelems, [&](int64_t i) { ker(i, i, i); });
    } else {
        const auto broadcast_mask = prb->get_broadcast_mask();

        dnnl::impl::parallel(0, [&](const int ithr, const int nthr) {
            int64_t start {0}, end {0};
            dnnl::impl::balance211(wei_nelems, nthr, ithr, start, end);
            if (start == end) return;

            for (int64_t i = 0; i < src_nelems; ++i) {
                const auto wei_idx = diff_src.get_scale_idx(i, broadcast_mask);
                if (wei_idx < start || wei_idx >= end) continue;
                ker(i, wei_idx, wei_idx);
            }
        });
    }
}

} // namespace prelu
