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

#include "tests/test_thread.hpp"

#include "conv/deconv.hpp"

namespace deconv {

void compute_ref_fwd(
        const conv::prb_t *prb, dnnl_primitive_t prim_ref, const args_t &args) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    // Swap arguments to re-use existing conv ref implementation.
    args_t ref_conv_args;
    for (int i = 0; i < args.size(); i++) {
        if (args.arg(i) == DNNL_ARG_SRC)
            ref_conv_args.set(DNNL_ARG_DIFF_DST, args.dnn_mem(i));
        else if (args.arg(i) == DNNL_ARG_WEIGHTS)
            ref_conv_args.set(
                    DNNL_ARG_WEIGHTS, args.find(DNNL_ARG_DIFF_WEIGHTS));
        else if (args.arg(i) == DNNL_ARG_DST)
            ref_conv_args.set(DNNL_ARG_DIFF_SRC, args.dnn_mem(i));
        else
            ref_conv_args.set(args.arg(i), args.dnn_mem(i));
    }

    using namespace conv;
    if (prb->alg == WINO && prb->cfg[SRC].dt == dnnl_f32) {
        compute_wino_ref_bwd_d(prb, ref_conv_args);
    } else {
        compute_ref_direct_bwd_d(prb, ref_conv_args);
    }
}

void compute_ref_bwd_d(
        const conv::prb_t *prb, dnnl_primitive_t prim_ref, const args_t &args) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    // Swap arguments to re-use existing conv ref implementation.
    args_t ref_conv_args;
    for (int i = 0; i < args.size(); i++) {
        if (args.arg(i) == DNNL_ARG_DIFF_SRC)
            ref_conv_args.set(DNNL_ARG_DST, args.dnn_mem(i));
        else if (args.arg(i) == DNNL_ARG_WEIGHTS)
            ref_conv_args.set(
                    DNNL_ARG_WEIGHTS, args.find(DNNL_ARG_DIFF_WEIGHTS));
        else if (args.arg(i) == DNNL_ARG_DIFF_DST)
            ref_conv_args.set(DNNL_ARG_SRC, args.dnn_mem(i));
        else
            ref_conv_args.set(args.arg(i), args.dnn_mem(i));
    }

    using namespace conv;
    if (prb->alg == WINO && prb->cfg[SRC].dt == dnnl_f32) {
        compute_wino_ref_fwd(prb, ref_conv_args);
    } else {
        compute_ref_direct_fwd(prb, ref_conv_args);
    }
}

void compute_ref_bwd_w(
        const conv::prb_t *prb, dnnl_primitive_t prim_ref, const args_t &args) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    // Swap arguments to re-use existing conv ref implementation.
    args_t ref_conv_args;
    for (int i = 0; i < args.size(); i++) {
        if (args.arg(i) == DNNL_ARG_SRC)
            ref_conv_args.set(DNNL_ARG_DIFF_DST, args.dnn_mem(i));
        else if (args.arg(i) == DNNL_ARG_DIFF_WEIGHTS)
            ref_conv_args.set(
                    DNNL_ARG_DIFF_WEIGHTS, args.find(DNNL_ARG_WEIGHTS));
        else if (args.arg(i) == DNNL_ARG_DIFF_DST)
            ref_conv_args.set(DNNL_ARG_SRC, args.dnn_mem(i));
        else
            ref_conv_args.set(args.arg(i), args.dnn_mem(i));
    }

    using namespace conv;
    if (prb->alg == WINO && prb->cfg[SRC].dt == dnnl_f32) {
        compute_wino_ref_bwd_w(prb, ref_conv_args);
    } else {
        compute_ref_bwd_weights(prb, ref_conv_args);
    }

    // Need to transpose data in weights back for proper comparison. This step
    // is done here as it's not needed for fast-ref-gpu.
    transpose_data_wei(
            prb, args.find(DNNL_ARG_WEIGHTS), args.find(DNNL_ARG_DIFF_WEIGHTS));

    // We don't reuse `compute_ref_bwd_bias` as it doesn't match arguments and
    // entry problem which is transposed - `p_tr`. Simpler to use the kernel
    // directly.
    // Take original memories, not `ref_conv_args`.
    if (prb->dir & FLAG_BIA) {
        const dnn_mem_t &diff_bia_m = args.find(DNNL_ARG_DIFF_BIAS);
        const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);
        /* help compiler optimize the code */
        const int64_t MB = prb->mb, G = prb->g;
        const int64_t OC = prb->ic; // prb.oc = p_tr.ic
        const int64_t OCG = OC / G;
        const int64_t OD = prb->id; // prb.od = p_tr.id
        const int64_t OH = prb->ih; // prb.oh = p_tr.ih
        const int64_t OW = prb->iw; // prb.ow = p_tr.iw

        dnnl::impl::parallel_nd(G, OCG, [&](int64_t g, int64_t oc) {
            size_t bia_off = g * OCG + oc;
            double sum = 0;

            for_(int64_t mb = 0; mb < MB; ++mb)
            for_(int64_t od = 0; od < OD; ++od)
            for_(int64_t oh = 0; oh < OH; ++oh)
            for (int64_t ow = 0; ow < OW; ++ow) {
                // src_off_f instead of dst_off_f due to inverse descriptor.
                size_t dst_off = src_off_f(prb, mb, g, oc, od, oh, ow);
                sum += ((float *)diff_dst_m)[dst_off];
            }
            ((float *)diff_bia_m)[bia_off] = (float)sum;
        });
    }
}

} // namespace deconv
