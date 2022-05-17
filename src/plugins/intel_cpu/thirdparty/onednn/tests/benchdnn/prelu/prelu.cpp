/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/compare.hpp"

#include "prelu/prelu.hpp"

namespace prelu {

int fill_data(data_kind_t kind, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Do fixed partitioning to have same filling for any number of threads.
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    dnnl::impl::parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note 1: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // we avoid it for two reasons:
        //   a. it has a complexity in O(idx_start).
        //   b. igen and fgen below might require more than 1 sample
        //   per idx, so the we cannot deterministically compute the
        //   number of states we need to discard
        // Note 2: We also advance the state to avoid having only
        // small values as first chunk input.  The +1 is necessary to
        // avoid generating zeros in first chunk.
        // Note 3: we multiply by kind + 1 to have different values in
        // src/dst and diff_dst. The +1 is to avoid 0 again.
        std::minstd_rand msr((idx_start + 1) * (kind + 1));
        msr.discard(1);
        std::uniform_int_distribution<> igen_02(0, 2), igen_05(0, 5);
        std::uniform_real_distribution<> fgen(-1.f, 1.f);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value;
            if (is_integral_dt(mem_dt.dt()))
                value = igen_05(msr);
            else
                value = kind == SRC
                        ? igen_02(msr)
                        : kind == WEI ? fgen(msr) : igen_02(msr) / 16.f;
            // TODO: amount of negative values should depend on number of points
            // to reduce as summation becomes inaccurate.
            float sign = mem_dt.dt() == dnnl_u8
                    ? 1.f
                    : flip_coin(idx, 0.1f) ? -1.f : 1.f;
            value = round_to_nearest_representable(mem_dt.dt(), sign * value);
            mem_fp.set_elem(idx, value);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int setup_prelu_po(const_dnnl_primitive_desc_t pd, std::vector<int> &args,
        std::vector<dnn_mem_t> &ref_mem, std::vector<dnn_mem_t> &prim_mem,
        const dnnl_engine_t &ref_engine) {
    const_dnnl_primitive_attr_t const_attr;
    DNN_SAFE(dnnl_primitive_desc_get_attr(pd, &const_attr), WARN);

    const_dnnl_post_ops_t const_attr_po;
    DNN_SAFE(
            dnnl_primitive_attr_get_post_ops(const_attr, &const_attr_po), WARN);

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(pd, dnnl_query_exec_arg_md, index);
    };

    const auto &dst_md = q(DNNL_ARG_DST);

    const auto po_len = dnnl_post_ops_len(const_attr_po);
    for (int idx = 0; idx < po_len; ++idx) {
        const auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
        if (kind != dnnl_prelu) continue;

        const auto ndims = dst_md.ndims;
        int mask = 0;
        dnnl_dims_t dims = {0};
        dnnl_post_ops_get_params_prelu(const_attr_po, idx, &mask);

        // Deduce prelu weights dims based on input policy.
        for (int d = 0; d < ndims; ++d) {
            dims[d] = (mask & (1 << d)) ? dst_md.dims[d] : 1;
        }

        // Following call can not be executed if po_md has runtime dimension due
        // to undefined size.
        ref_mem.emplace_back(ndims, dims, dnnl_f32, tag::abx, ref_engine);
        prim_mem.emplace_back(
                ndims, dims, dnnl_f32, tag::axb, get_test_engine());
        args.push_back(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_WEIGHTS);
        fill_data(WEI, prim_mem.back(), ref_mem.back());
    }
    return OK;
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &ppd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_prelu_desc_t pd;
    dnnl_memory_desc_t data_d, weights_d;

    const auto &src_dims = prb->sdims[0];
    const auto &weight_dims = prb->sdims[1];

    SAFE(init_md(&data_d, prb->ndims, src_dims.data(), prb->sdt[0],
                 prb->stag[0]),
            CRIT);
    SAFE(init_md(&weights_d, prb->ndims, weight_dims.data(), prb->sdt[1],
                 prb->stag[1]),
            CRIT);

    if (prb->dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        DNN_SAFE(dnnl_prelu_forward_desc_init(&pd, prop, &data_d, &weights_d),
                WARN);
    } else {
        dnnl_memory_desc_t diff_data_d, diff_weights_d;
        SAFE(init_md(&diff_data_d, prb->ndims, src_dims.data(), prb->sdt[0],
                     prb->stag[0]),
                CRIT);
        SAFE(init_md(&diff_weights_d, prb->ndims, weight_dims.data(),
                     prb->sdt[1], prb->stag[1]),
                CRIT);

        DNN_SAFE(dnnl_prelu_backward_desc_init(&pd, &data_d, &weights_d,
                         &diff_data_d, &diff_weights_d),
                WARN);
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&ppd, &pd, dnnl_attr, engine, nullptr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    res->impl_name = query_impl_info(ppd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());

    SAFE(check_pd_w_and_wo_attr(res, prb->attr, pd), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common(prb->sdt, FWD_D, res);
    if (res->state == SKIPPED) return;

    if (is_nvidia_gpu()) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    check_sum_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(prim, &const_pd), CRIT);

    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &data_md = q(DNNL_ARG_SRC);
    const auto &weight_md = q(DNNL_ARG_WEIGHTS);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);
    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(data_md, dnnl_f32, tag::abx, test_engine);
    dnn_mem_t weights_fp(weight_md, dnnl_f32, tag::abx, test_engine);

    dnn_mem_t src_dt(data_md, test_engine);
    dnn_mem_t weights_dt(weight_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    SAFE(fill_data(SRC, src_dt, src_fp), WARN);
    SAFE(fill_data(WEI, weights_dt, weights_fp), WARN);

    args_t args;
    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_WEIGHTS, weights_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    dnn_mem_t dst_dt, d_src_fp, d_src_dt, d_dst_fp, d_dst_dt, d_weights_fp,
            d_weights_dt;

    if (prb->dir & FLAG_FWD) {
        dnn_mem_t dst_fp(data_md, dnnl_f32, tag::abx, test_engine);
        dst_dt = dnn_mem_t(data_md, test_engine);

        args.set(DNNL_ARG_DST, dst_dt);
        SAFE(execute_and_wait(prim, args), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_fwd(prb, src_fp, weights_fp, dst_fp);
            compare::compare_t cmp;
            cmp.set_threshold(2 * epsilon_dt(prb->sdt[0]));
            cmp.set_zero_trust_percent(50.f); // Due to filling
            cmp.set_data_kind(DST);
            SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
        }
    } else {
        const auto &d_data_md = q(DNNL_ARG_DIFF_DST);
        const auto &d_weights_md = q(DNNL_ARG_DIFF_WEIGHTS);

        dnn_mem_t d_src_fp(d_data_md, dnnl_f32, tag::abx, test_engine);
        dnn_mem_t d_weights_fp(d_weights_md, dnnl_f32, tag::abx, test_engine);
        dnn_mem_t d_dst_fp(d_data_md, dnnl_f32, tag::abx, test_engine);

        d_src_dt = dnn_mem_t(d_data_md, test_engine);
        d_weights_dt = dnn_mem_t(d_weights_md, test_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        SAFE(fill_data(DST, d_dst_dt, d_dst_fp), WARN);

        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, d_weights_dt);
        SAFE(execute_and_wait(prim, args), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_bwd(
                    prb, src_fp, weights_fp, d_src_fp, d_dst_fp, d_weights_fp);

            compare::compare_t cmp_src;
            cmp_src.set_threshold(2 * epsilon_dt(prb->sdt[0]));
            cmp_src.set_zero_trust_percent(50.f); // Due to filling
            cmp_src.set_data_kind(SRC);
            SAFE(cmp_src.compare(d_src_fp, d_src_dt, prb->attr, res), WARN);

            compare::compare_t cmp_wei;
            cmp_wei.set_threshold(2 * epsilon_dt(prb->sdt[1]));
            // Weights are very sparse, no sense to test for trust.
            cmp_wei.set_zero_trust_percent(100.f);
            cmp_wei.set_data_kind(WEI);
            SAFE(cmp_wei.compare(d_weights_fp, d_weights_dt, prb->attr, res),
                    WARN);
        }
    }

    return measure_perf(res->timer, prim, args);
}

} // namespace prelu
