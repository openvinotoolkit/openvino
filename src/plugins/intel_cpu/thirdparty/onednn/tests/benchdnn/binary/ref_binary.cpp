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

#include "binary/binary.hpp"

namespace binary {

void compute_ref(const prb_t *prb, const dnn_mem_t &src0, const dnn_mem_t &src1,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst) {
    float *dst_ptr = (float *)dst;
    const float *A = (const float *)src0;
    const float *B = (const float *)src1;

    dims_t ddims(dst.md_);

    float scales[2] = {prb->attr.scales.get(DNNL_ARG_SRC_0).scale,
            prb->attr.scales.get(DNNL_ARG_SRC_1).scale};

    const auto nelems = dst.nelems();
    const auto broadcast_mask_A = prb->get_broadcast_mask(ddims, 0);
    const auto broadcast_mask_B = prb->get_broadcast_mask(ddims, 1);
    std::vector<int> v_bin_po_mask = prb->attr.post_ops.get_binary_po_masks();

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const auto idx_A = dst.get_scale_idx(i, broadcast_mask_A);
        const auto idx_B = dst.get_scale_idx(i, broadcast_mask_B);
        float res = compute_binary(
                prb->alg, scales[0] * A[idx_A], scales[1] * B[idx_B]);
        float &dst_fp = dst_ptr[i];

        std::vector<float> v_binary_vals(v_bin_po_mask.size());
        for (size_t d = 0; d < v_bin_po_mask.size(); ++d) {
            const auto bin_po_offset = dst.get_scale_idx(i, v_bin_po_mask[d]);
            const float binary_val = binary_po[d].get_elem(bin_po_offset);
            v_binary_vals[d] = binary_val;
        }
        maybe_post_ops(prb->attr, res, maybe_saturate(prb->ddt, dst_fp),
                v_binary_vals);
        maybe_saturate(prb->ddt, res);
        dst_fp = res;
    });
}

} // namespace binary
