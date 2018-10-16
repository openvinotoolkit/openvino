/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "ip/ip.hpp"

namespace ip {

inline int init_pd(const prb_t *p, mkldnn_inner_product_desc_t &ipd,
        mkldnn_primitive_desc_t &ippd) {
    mkldnn_memory_desc_t src_d, wei_d, bia_d, dst_d;

    mkldnn_dims_t src_dims = {p->mb, p->ic, p->ih, p->iw};
    mkldnn_dims_t wei_dims = {p->oc, p->ic, p->ih, p->iw};
    mkldnn_dims_t bia_dims = {p->oc};
    mkldnn_dims_t dst_dims = {p->mb, p->oc};

    DNN_SAFE(mkldnn_memory_desc_init(&src_d, 4, src_dims, p->src_dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&wei_d, 4, wei_dims, p->wei_dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&bia_d, 1, bia_dims, p->dst_dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&dst_d, 2, dst_dims, p->dst_dt, mkldnn_any), WARN);

    switch (p->dir) {
    case FWD_D: case FWD_B:
        DNN_SAFE(mkldnn_inner_product_forward_desc_init(&ipd, mkldnn_forward,
                    &src_d, &wei_d, p->dir == FWD_D ? NULL : &bia_d, &dst_d),
                WARN);
        break;
    case BWD_D:
        DNN_SAFE(mkldnn_inner_product_backward_data_desc_init(&ipd, &src_d,
                    &wei_d, &dst_d), WARN);
        break;
    case BWD_W: case BWD_WB:
        DNN_SAFE(mkldnn_inner_product_backward_weights_desc_init(&ipd, &src_d,
                    &wei_d, p->dir == BWD_W ? NULL : &bia_d, &dst_d), WARN);
        break;
    default: DNN_SAFE(mkldnn_invalid_arguments, CRIT);
    }

    DNN_SAFE(ipd.accum_data_type == p->acc_dt
            ? mkldnn_success : mkldnn_unimplemented, CRIT);

    DNN_SAFE(mkldnn_primitive_desc_create(&ippd, &ipd, engine, NULL), WARN);

    auto q = [=](mkldnn_query_t query, int index = 0) {
        return *mkldnn_primitive_desc_query_memory_d(
                mkldnn_primitive_desc_query_pd(ippd, query, index));
    };

    if (p->dir == BWD_D)
        ipd.diff_src_desc = q(mkldnn_query_diff_src_pd);
    else
        ipd.src_desc = q(mkldnn_query_src_pd);

    if (p->dir & FLAG_WEI)
        ipd.diff_weights_desc = q(mkldnn_query_diff_weights_pd);
    else
        ipd.weights_desc = q(mkldnn_query_weights_pd);

    if (p->dir & FLAG_BIA) {
        if (p->dir & FLAG_BWD)
            ipd.diff_bias_desc = q(mkldnn_query_diff_weights_pd, 1);
        else
            ipd.bias_desc = q(mkldnn_query_weights_pd, 1);
    }

    if (p->dir & FLAG_BWD)
        ipd.diff_dst_desc = q(mkldnn_query_diff_dst_pd);
    else
        ipd.dst_desc = q(mkldnn_query_dst_pd);

    return OK;
}

inline int compare_dat(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    size_t nelems = mem_dt.nelems();
    double eps = 1e-4;

    r->errors = 0;
    r->total = nelems;
    float max_rel_diff = 0;
    for (size_t i = 0; i < nelems; ++i) {
        float dt = ((float*)mem_dt)[i];
        float fp = ((float*)mem_fp)[i];
        float diff = fabsf(fp - dt);
        float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        int ok = (fabs(fp) > 1e-5 ? rel_diff : diff) < eps;
        if (!ok) {
            r->errors++;
            if (max_rel_diff < rel_diff) max_rel_diff = rel_diff;
            if (r->errors < 10)
                printf("[%4d] fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                        (int)i, fp, dt, diff, rel_diff);
        }
    }

    return r->errors ? FAIL : OK;
}

inline void fill_src(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    dnn_mem_t mem_00(mem_dt.md_, mkldnn_f32, mkldnn_nchw);
    const ptrdiff_t sz = (ptrdiff_t)mem_00.nelems();
#   pragma omp parallel for
    for (ptrdiff_t i = 0; i < sz; ++i)
        ((float*)mem_00)[i] = 1 + (i % 3); // 1 + sin(0.2* (i % 17));

    mem_dt.reorder(mem_00);
    mem_fp.reorder(mem_dt);

    int sanity_err = compare_dat(mem_fp, mem_00, r);
    if (sanity_err != OK) {
        printf("@@@ sanity failed: %s:%d\n", __func__, __LINE__);
    }
}

inline void fill_wei(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    dnn_mem_t mem_00(mem_dt.md_, mkldnn_f32, mkldnn_oihw);
    const ptrdiff_t sz = (ptrdiff_t)mem_00.nelems();
#   pragma omp parallel for
    for (ptrdiff_t i = 0; i < sz; ++i)
        ((float*)mem_00)[i] = (i % 4) - 1 ; // 1 + sin(0.2* (i % 17));

    mem_dt.reorder(mem_00);
    mem_fp.reorder(mem_dt);

    int sanity_err = compare_dat(mem_fp, mem_00, r);
    if (sanity_err != OK) {
        printf("@@@ sanity failed: %s:%d\n", __func__, __LINE__);
    }
}

inline void fill_bia(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    dnn_mem_t mem_00(mem_dt.md_, mkldnn_f32, mkldnn_x);
    const size_t sz = mem_00.nelems();
    for (size_t i = 0; i < sz; ++i)
        ((float*)mem_00)[i] = 0; // 1 + sin(0.2* (i % 17));

    mem_dt.reorder(mem_00);
    mem_fp.reorder(mem_dt);

    int sanity_err = compare_dat(mem_fp, mem_00, r);
    if (sanity_err != OK) {
        printf("@@@ sanity failed: %s:%d\n", __func__, __LINE__);
    }
}

inline void fill_dst(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    dnn_mem_t mem_00(mem_dt.md_, mkldnn_f32, mkldnn_nc);
    const ptrdiff_t sz = (ptrdiff_t)mem_00.nelems();
#   pragma omp parallel for
    for (ptrdiff_t i = 0; i < sz; ++i)
        ((float*)mem_00)[i] = 1 + (i % 3); // 1 + sin(0.2* (i % 17));

    mem_dt.reorder(mem_00);
    mem_fp.reorder(mem_dt);

    int sanity_err = compare_dat(mem_fp, mem_00, r);
    if (sanity_err != OK) {
        printf("@@@ sanity failed: %s:%d\n", __func__, __LINE__);
    }
}

int doit(prb_t *p, res_t *r) {
    mkldnn_inner_product_desc_t ipd;
    mkldnn_primitive_desc_t ippd;
    mkldnn_primitive_t ip;

    SAFE(init_pd(p, ipd, ippd), WARN);

    auto &src_dt_d = p->dir == BWD_D ? ipd.diff_src_desc : ipd.src_desc;
    auto &wei_dt_d = p->dir & FLAG_WEI ? ipd.diff_weights_desc : ipd.weights_desc;
    auto &bia_dt_d = p->dir & FLAG_BWD ? ipd.diff_bias_desc : ipd.bias_desc;
    auto &dst_dt_d = p->dir & FLAG_BWD ? ipd.diff_dst_desc: ipd.dst_desc;

    const auto fp = mkldnn_f32;
    dnn_mem_t src_dt(src_dt_d, p->src_dt);
    dnn_mem_t wei_dt(wei_dt_d, p->wei_dt);
    dnn_mem_t dst_dt(dst_dt_d, p->dst_dt);
    dnn_mem_t bia_dt = p->dir & FLAG_BIA
        ? dnn_mem_t(bia_dt_d, p->dst_dt) : dnn_mem_t();

    dnn_mem_t src_fp(src_dt_d, fp, mkldnn_nchw);
    dnn_mem_t wei_fp(wei_dt_d, fp, mkldnn_oihw);
    dnn_mem_t dst_fp(dst_dt_d, fp, mkldnn_nc);
    dnn_mem_t bia_fp = p->dir & FLAG_BIA
        ? dnn_mem_t(bia_dt_d, fp, mkldnn_x) : dnn_mem_t();

    fill_src(src_dt, src_fp, r);
    fill_wei(wei_dt, wei_fp, r);
    fill_dst(dst_dt, dst_fp, r);
    if (p->dir & FLAG_BIA)
        fill_bia(bia_dt, bia_fp, r);

    if (p->dir & FLAG_FWD) {
        compute_ref_fwd(p, src_fp, wei_fp, bia_fp, dst_fp);
        mkldnn_primitive_at_t inputs[3] = { {src_dt.p_, 0}, {wei_dt.p_, 0},
            {p->dir & FLAG_BIA ? bia_dt.p_ : NULL, 0}
        };
        const_mkldnn_primitive_t outputs[] = { dst_dt.p_ };
        DNN_SAFE(mkldnn_primitive_create(&ip, ippd, inputs, outputs), WARN);
        execute(ip);
        dnn_mem_t dst(dst_dt, fp, mkldnn_nc);
        dst.reorder(dst_dt);
        compare_dat(dst, dst_fp, r);
    } else if (p->dir == BWD_D) {
#if 0
        compute_ref_bwd_d(p, src_fp, wei_fp, dst_fp);
        mkldnn_primitive_at_t inputs[3] = { {dst_dt.p_, 0}, {wei_dt.p_, 0}, };
        const_mkldnn_primitive_t outputs[] = { src_dt.p_ };
        DNN_SAFE(mkldnn_primitive_create(&c, ippd, inputs, outputs), WARN);
        execute(c);
        dnn_mem_t src(src_dt, fp, mkldnn_nchw);
        src.reorder(src_dt);
        compare_dat(src, src_fp, r);
    } else if (p->dir & FLAG_BWD && p->dir & FLAG_WEI) {
        compute_ref_bwd_w(p, src_fp, wei_fp, bia_fp, dst_fp);
        mkldnn_primitive_at_t inputs[3] = { {src_dt.p_, 0}, {dst_dt.p_, 0}, };
        const_mkldnn_primitive_t outputs[] = { wei_dt.p_,
            p->dir & FLAG_BIA ? bia_dt.p_ : NULL,
        };
        DNN_SAFE(mkldnn_primitive_create(&c, ippd, inputs, outputs), WARN);
        execute(c);
        dnn_mem_t wei(wei_dt, fp, mkldnn_goihw);
        wei.reorder(wei_dt);
        if (compare_dat(wei, wei_fp, r) != 0) return FAIL;
        if (p->dir & FLAG_BIA) {
            dnn_mem_t bia(bia_dt, fp, mkldnn_x);
            bia.reorder(bia_dt);
            compare_dat(bia, bia_fp, r);
        }
#endif
    }

    return OK;
}

}
