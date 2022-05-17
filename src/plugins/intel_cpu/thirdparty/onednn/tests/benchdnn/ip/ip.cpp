/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#include <cstring>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "ip/ip.hpp"

namespace ip {

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &ippd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_inner_product_desc_t ipd;
    dnnl_memory_desc_t src_d, wei_d, bia_d, dst_d;

    dnnl_dims_t src_dims_0d = {prb->mb, prb->ic};
    dnnl_dims_t src_dims_1d = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t src_dims_2d = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t src_dims_3d = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};
    dnnl_dims_t wei_dims_0d = {prb->oc, prb->ic};
    dnnl_dims_t wei_dims_1d = {prb->oc, prb->ic, prb->iw};
    dnnl_dims_t wei_dims_2d = {prb->oc, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t wei_dims_3d = {prb->oc, prb->ic, prb->id, prb->ih, prb->iw};
    dnnl_dims_t bia_dims = {prb->oc};
    dnnl_dims_t dst_dims = {prb->mb, prb->oc};

    dnnl_dim_t *src_dims = prb->ndims == 5
            ? src_dims_3d
            : prb->ndims == 4 ? src_dims_2d
                              : prb->ndims == 3 ? src_dims_1d : src_dims_0d;

    dnnl_dim_t *wei_dims = prb->ndims == 5
            ? wei_dims_3d
            : prb->ndims == 4 ? wei_dims_2d
                              : prb->ndims == 3 ? wei_dims_1d : wei_dims_0d;

    SAFE(init_md(&src_d, prb->ndims, src_dims, prb->cfg[SRC].dt, prb->stag),
            CRIT);
    SAFE(init_md(&wei_d, prb->ndims, wei_dims, prb->cfg[WEI].dt, prb->wtag),
            CRIT);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&bia_d, 1, bia_dims, prb->cfg[BIA].dt,
                     dnnl_format_tag_any),
            WARN);
    SAFE(init_md(&dst_d, 2, dst_dims, prb->cfg[DST].dt, prb->dtag), CRIT);

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_inner_product_forward_desc_init(&ipd,
                             prb->dir == FWD_I ? dnnl_forward_inference
                                               : dnnl_forward_training,
                             &src_d, &wei_d,
                             prb->dir == FWD_B ? &bia_d : nullptr, &dst_d),
                    WARN);
            break;
        case BWD_D:
            DNN_SAFE(dnnl_inner_product_backward_data_desc_init(
                             &ipd, &src_d, &wei_d, &dst_d),
                    WARN);
            break;
        case BWD_W:
        case BWD_WB:
            DNN_SAFE(dnnl_inner_product_backward_weights_desc_init(&ipd, &src_d,
                             &wei_d, prb->dir == BWD_W ? nullptr : &bia_d,
                             &dst_d),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(ipd.accum_data_type == prb->cfg[ACC].dt ? dnnl_success
                                                     : dnnl_unimplemented,
            CRIT);

    attr_args_t attr_args;
    attr_args.prepare_output_scales(prb->attr, prb->scales, prb->oc);
    attr_args.prepare_post_ops_mds(prb->attr, 2, dst_dims);
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    dnnl_status_t init_status = dnnl_success;
    init_status = dnnl_primitive_desc_create(
            &ippd, &ipd, dnnl_attr, engine, nullptr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(ippd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    SAFE(check_pd_w_and_wo_attr(res, prb->attr, ipd), WARN);

    return OK;
}

bool need_src_init(const prb_t *prb) {
    return !(prb->dir == BWD_D);
}

bool need_wei_init(const prb_t *prb) {
    return !(prb->dir & FLAG_BWD && prb->dir & FLAG_WEI);
}

bool need_bia_init(const prb_t *prb) {
    return need_wei_init(prb);
}

bool need_dst_init(const prb_t *prb) {
    return !(prb->dir & FLAG_FWD)
            || (prb->attr.post_ops.find(attr_t::post_ops_t::SUM) >= 0);
}

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto &c = prb->cfg[SRC];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(prb->mb, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t mb, int64_t ic, int64_t id, int64_t ih, int64_t iw) {
                const int gen
                        = 101 * id + 103 * ih + 107 * iw + 109 * mb + 113 * ic;
                const float sparsity = prb->ic < 5 ? 1.f : c.f_sparsity;
                const bool non_base = flip_coin(gen, sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;
                ((float *)mem_fp)[src_off_f(prb, mb, ic, id, ih, iw)] = value;
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_wei(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool s8_s8
            = prb->cfg[WEI].dt == dnnl_s8 && prb->cfg[SRC].dt == dnnl_s8;

    const auto &c = prb->cfg[WEI];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(prb->oc, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t oc, int64_t ic, int64_t kd, int64_t kh, int64_t kw) {
                const int gen = 127 * kd + 131 * kh + 137 * kw + 139 * oc
                        + 149 * ic + 7;
                const float sparsity = prb->ic < 5 ? 1.f : c.f_sparsity;
                const bool non_base = flip_coin(gen, sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;
                ((float *)mem_fp)[wei_off_f(prb, oc, ic, kd, kh, kw)] = value;
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);
    if (s8_s8 && is_cpu()) {
        // Check that s8 -> s8_comp exists in the library since users may have
        // already quantized data.
        dnn_mem_t mem_fp_s8(mem_fp.md_, dnnl_s8, get_test_engine());
        dnn_mem_t mem_dt_s8(mem_dt.md_, dnnl_s8, get_test_engine());
        SAFE(mem_fp_s8.reorder(mem_fp), WARN);
        SAFE(mem_dt_s8.reorder(mem_fp_s8), WARN);
        SAFE(mem_dt.size() == mem_dt_s8.size() ? OK : FAIL, WARN);
        int rc = std::memcmp((void *)mem_dt, (void *)mem_dt_s8, mem_dt.size());
        SAFE(rc == 0 ? OK : FAIL, WARN);
    }

    return OK;
}

int fill_bia(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto &c = prb->cfg[BIA];
    const int range = c.f_max - c.f_min + 1;

    for (size_t i = 0; i < nelems; ++i) {
        const int gen = (int)(151 * i + 11);
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value = non_base ? c.f_min + gen * 1 % range : c.f_base;
        ((float *)mem_fp)[i] = value;
    }

    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto &c = prb->cfg[DST];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(prb->mb, prb->oc, [&](int64_t mb, int64_t oc) {
        const int gen = 173 * mb + 179 * oc;
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value = non_base ? c.f_min + gen * 1 % range : c.f_base;

        ((float *)mem_fp)[dst_off_f(prb, mb, oc)] = value;
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common(
            {prb->cfg[SRC].dt, prb->cfg[WEI].dt, prb->cfg[DST].dt}, prb->dir,
            res);
    if (res->state == SKIPPED) return;

    if (is_nvidia_gpu()) {
        const auto &po = prb->attr.post_ops;
        bool post_ops_ok = true;
        for (int i = 0; i < po.len(); ++i) {
            const auto &e = po.entry[i];
            if (e.is_sum_kind())
                continue;
            else if (e.is_eltwise_kind())
                post_ops_ok = post_ops_ok && is_nvidia_eltwise_ok(prb->dir, e);
            else if (e.is_binary_kind() || e.is_convolution_kind())
                post_ops_ok = false;
            else
                assert(!"unknown post-op type");
        }

        const bool oscale_ok = prb->attr.oscale.policy == policy_t::COMMON;

        if (!post_ops_ok || !oscale_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
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

    const auto &src_md
            = prb->dir == BWD_D ? q(DNNL_ARG_DIFF_SRC) : q(DNNL_ARG_SRC);
    const auto &wei_md = prb->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_WEIGHTS)
                                             : q(DNNL_ARG_WEIGHTS);
    const auto &bia_md
            = prb->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_BIAS) : q(DNNL_ARG_BIAS);
    const auto &dst_md
            = prb->dir & FLAG_BWD ? q(DNNL_ARG_DIFF_DST) : q(DNNL_ARG_DST);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto src_tag = tag::abx;
    const auto wei_tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t bia_dt(bia_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    dnn_mem_t src_fp(src_md, fp, src_tag, test_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, test_engine);
    dnn_mem_t bia_fp(bia_md, fp, tag::x, test_engine);
    dnn_mem_t dst_fp(dst_md, fp, tag::abx, test_engine);

    if (need_src_init(prb)) SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    if (need_wei_init(prb)) SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    if (need_bia_init(prb)) SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);
    if (need_dst_init(prb)) SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    args_t args;

    if (prb->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        args.set(binary_po_args, binary_po_dt);

        SAFE(execute_and_wait(prim, args), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_fwd(test_engine, prb, src_fp, wei_fp, bia_fp,
                    binary_po_fp, dst_fp);
            compare::compare_t cmp;
            cmp.set_threshold(prb->cfg[DST].eps);
            cmp.set_data_kind(DST);
            cmp.set_zero_trust_percent(80.f); // TODO: why so bad filling?
            SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
        }
    } else if (prb->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_bwd_d(prb, src_fp, wei_fp, dst_fp);
            compare::compare_t cmp;
            cmp.set_threshold(prb->cfg[SRC].eps);
            cmp.set_data_kind(SRC);
            cmp.set_zero_trust_percent(80.f); // TODO: why so bad filling?
            SAFE(cmp.compare(src_fp, src_dt, prb->attr, res), WARN);
        }
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_BIAS, bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_bwd_w(prb, src_fp, wei_fp, bia_fp, dst_fp);
            compare::compare_t cmp;
            cmp.set_threshold(prb->cfg[WEI].eps);
            cmp.set_data_kind(WEI);
            cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?
            SAFE(cmp.compare(wei_fp, wei_dt, prb->attr, res), WARN);
            if (prb->dir & FLAG_BIA) {
                cmp.set_threshold(prb->cfg[BIA].eps);
                cmp.set_data_kind(BIA);
                cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?
                SAFE(cmp.compare(bia_fp, bia_dt, prb->attr, res), WARN);
            }
        }
    }

    return measure_perf(res->timer, prim, args);
}

} // namespace ip
