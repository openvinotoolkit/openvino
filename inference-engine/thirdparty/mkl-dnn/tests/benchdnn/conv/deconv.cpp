/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "src/common/mkldnn_thread.hpp"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "norm.hpp"

#include "conv/deconv.hpp"
using namespace conv;

namespace deconv {

inline static void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}
inline bool is_deconv_3d(const prb_t *p)
{
    return (p->id > 1 || p->od > 1) ? 1 : 0;
}

inline int transpose_data_wei(const prb_t *p, dnn_mem_t &wei, dnn_mem_t &wei_tr) {
    mkldnn::impl::parallel_nd(
        p->g, p->oc / p->g, p->ic / p->g, p->kd, p->kh, p->kw,
        [&](int g, int oc, int ic, int kd, int kh, int kw) {
        size_t idx = (((((size_t)g * p->ic / p->g + ic) * p->oc / p->g + oc)
        * p->kd + kd) * p->kh + kh) * p->kw + kw;
        ((float*)wei_tr)[idx] = ((float*)wei)[wei_off_f(p, g, oc, ic, kd, kh, kw)];
    });

    return OK;
}

inline int init_pd(const prb_t *p, mkldnn_deconvolution_desc_t &cd,
        mkldnn_primitive_desc_t &dpd, res_t *r) {
    int ndims = is_deconv_3d(p) ? 5 : 4;

    mkldnn_memory_desc_t src_d, wei_d, bia_d, dst_d;
    mkldnn_dims_t src_dims = {p->mb, p->ic, p->ih, p->iw};
    mkldnn_dims_t src_3d_dims = {p->mb, p->ic, p->id, p->ih, p->iw};
    mkldnn_dims_t wei_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kh, p->kw};
    mkldnn_dims_t wei_3d_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kd, p->kh, p->kw};
    mkldnn_dims_t bia_dims = {p->oc};
    mkldnn_dims_t dst_dims = {p->mb, p->oc, p->oh, p->ow};
    mkldnn_dims_t dst_3d_dims = {p->mb, p->oc, p->od, p->oh, p->ow};

    DNN_SAFE(mkldnn_memory_desc_init(&src_d, ndims,
        is_deconv_3d(p) ? src_3d_dims : src_dims, p->cfg[SRC].dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&wei_d, ndims + 1,
        is_deconv_3d(p) ? wei_3d_dims : wei_dims, p->cfg[WEI].dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&bia_d, 1, bia_dims, p->cfg[BIA].dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&dst_d, ndims,
        is_deconv_3d(p) ? dst_3d_dims : dst_dims, p->cfg[DST].dt, mkldnn_any), WARN);
    int strides_2d[] = {p->sh, p->sw};
    int dilates_2d[] = {p->dh, p->dw};
    int padding_2d[] = {p->ph, p->pw};
    int strides_3d[] = {p->sd, p->sh, p->sw};
    int dilates_3d[] = {p->dd, p->dh, p->dw};
    int padding_3d[] = {p->pd, p->ph, p->pw};

    auto bph = [&](int ih, int oh, int kh, int sh, int ph, int dh) {
        return (oh - 1) * sh - ih + ((kh - 1) * (dh + 1) + 1) - ph;
    };
    int padding_r_3d[] = {
        bph(p->od, p->id, p->kd, p->sd, p->pd, p->dd),
        bph(p->oh, p->ih, p->kh, p->sh, p->ph, p->dh),
        bph(p->ow, p->iw, p->kw, p->sw, p->pw, p->dw)};
    int padding_r_2d[] = {
        bph(p->oh, p->ih, p->kh, p->sh, p->ph, p->dh),
        bph(p->ow, p->iw, p->kw, p->sw, p->pw, p->dw)};

    int *strides = is_deconv_3d(p) ? strides_3d : strides_2d;
    int *dilates = is_deconv_3d(p) ? dilates_3d : dilates_2d;
    int *padding = is_deconv_3d(p) ? padding_3d : padding_2d;
    int *padding_r = is_deconv_3d(p) ? padding_r_3d : padding_r_2d;
    mkldnn_alg_kind_t alg = mkldnn_deconvolution_direct;
    if (p->alg == WINO) alg = mkldnn_deconvolution_winograd;

    switch (p->dir) {
    case FWD_D: case FWD_B:
        DNN_SAFE(mkldnn_dilated_deconvolution_forward_desc_init(&cd,
                    mkldnn_forward_inference, alg, &src_d, &wei_d,
                    p->dir == FWD_D ? NULL : &bia_d, &dst_d, strides,
                    dilates, padding, padding_r, mkldnn_padding_zero), WARN);
        break;
    case BWD_D:
        DNN_SAFE(mkldnn_dilated_deconvolution_backward_data_desc_init(&cd, alg,
                    &src_d, &wei_d, &dst_d, strides, dilates, padding,
                    padding_r, mkldnn_padding_zero), WARN);
        break;
    case BWD_W: case BWD_WB:
        DNN_SAFE(mkldnn_dilated_deconvolution_backward_weights_desc_init(&cd,
                    alg, &src_d, &wei_d, p->dir == BWD_W ? NULL : &bia_d,
                    &dst_d, strides, dilates,  padding, padding_r,
                    mkldnn_padding_zero), WARN);
        break;
    default: DNN_SAFE(mkldnn_invalid_arguments, CRIT);
    }

    DNN_SAFE(cd.accum_data_type == p->cfg[ACC].dt
            ? mkldnn_success : mkldnn_unimplemented, CRIT);

    auto mkldnn_attr = create_mkldnn_attr(p->attr, p->oc, p->scales);

    mkldnn_status_t init_status = mkldnn_success;
    init_status = mkldnn_primitive_desc_create_v2(&dpd, &cd, mkldnn_attr,
            engine, NULL);

    mkldnn_primitive_attr_destroy(mkldnn_attr);

    if (init_status == mkldnn_unimplemented)
    {
        return r->state = UNIMPLEMENTED, OK;
    } else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(dpd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: mkldnn implementation: %s\n", impl_str);
        DNN_SAFE(mkldnn_primitive_desc_destroy(dpd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "mkldnn implementation: %s\n", impl_str);
    }

    auto q = [=](mkldnn_query_t query, int index = 0) {
        return *mkldnn_primitive_desc_query_memory_d(
                mkldnn_primitive_desc_query_pd(dpd, query, index));
    };

    if (p->dir == BWD_D)
        cd.diff_src_desc = q(mkldnn_query_diff_src_pd);
    else
        cd.src_desc = q(mkldnn_query_src_pd);

    if (p->dir & FLAG_WEI)
        cd.diff_weights_desc = q(mkldnn_query_diff_weights_pd);
    else
        cd.weights_desc = q(mkldnn_query_weights_pd);

    if (p->dir & FLAG_BIA) {
        if (p->dir & FLAG_BWD)
            cd.diff_bias_desc = q(mkldnn_query_diff_weights_pd, 1);
        else
            cd.bias_desc = q(mkldnn_query_weights_pd, 1);
    }

    if (p->dir & FLAG_BWD)
        cd.diff_dst_desc = q(mkldnn_query_diff_dst_pd);
    else
        cd.dst_desc = q(mkldnn_query_dst_pd);

    return OK;
}
int doit(const prb_t *p, res_t *r) {
    res_t res_zero{};
    *r = res_zero;
    bool with_groups = 1;

    prb_t p_tr((desc_t)*p, p->dir, p->cfg, p->alg, p->merge, p->attr, p->mb);
    swap(p_tr.ic,  p_tr.oc);
    swap(p_tr.ih,  p_tr.oh);
    swap(p_tr.id,  p_tr.od);
    swap(p_tr.iw,  p_tr.ow);

    mkldnn_deconvolution_desc_t cd;
    mkldnn_primitive_desc_t dpd;
    mkldnn_primitive_t c{};

    SAFE(init_pd(p, cd, dpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
        return OK;

    auto &src_dt_d = p->dir == BWD_D ? cd.diff_src_desc : cd.src_desc;
    auto &wei_dt_d = p->dir & FLAG_WEI ? cd.diff_weights_desc : cd.weights_desc;
    auto &bia_dt_d = p->dir & FLAG_BWD ? cd.diff_bias_desc : cd.bias_desc;
    auto &dst_dt_d = p->dir & FLAG_BWD ? cd.diff_dst_desc: cd.dst_desc;
    auto wei_tr_dt_d = wei_dt_d;
    swap(wei_tr_dt_d.dims[with_groups+0], wei_tr_dt_d.dims[with_groups+1]);

    dnn_mem_t src_dt(src_dt_d, p->cfg[SRC].dt);
    dnn_mem_t wei_dt(wei_dt_d, p->cfg[WEI].dt);
    dnn_mem_t dst_dt(dst_dt_d, p->cfg[DST].dt);
    dnn_mem_t *p_bia_dt = p->dir & FLAG_BIA
        ? new dnn_mem_t(bia_dt_d, p->cfg[BIA].dt) : new dnn_mem_t();
    dnn_mem_t &bia_dt = *p_bia_dt;

    auto src_format = is_deconv_3d(p) ? mkldnn_ncdhw : mkldnn_nchw;
    auto wei_format = is_deconv_3d(p) ? mkldnn_goidhw : mkldnn_goihw;

    const auto fp = mkldnn_f32;

    /* memory for ref */
    dnn_mem_t src_fp(src_dt_d, fp, src_format);
    dnn_mem_t wei_fp(wei_dt_d, fp, wei_format);
    dnn_mem_t dst_fp(dst_dt_d, fp, src_format);
    dnn_mem_t wei_tr_fp(wei_tr_dt_d, fp, wei_format);
    dnn_mem_t *p_bia_fp = p->dir & FLAG_BIA
        ? new dnn_mem_t(bia_dt_d, fp, mkldnn_x) : new dnn_mem_t();
    dnn_mem_t *p_zero_fp = new dnn_mem_t();
    dnn_mem_t &bia_fp = *p_bia_fp, &zero_fp = *p_zero_fp;

    /* fill memory + reorders <-> */
    SAFE(fill_dst(p, dst_dt, dst_fp, r), WARN);
    SAFE(fill_wei(p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_src(p, src_dt, src_fp, r), WARN);

    SAFE(transpose_data_wei(p, wei_fp, wei_tr_fp), WARN);
    if (p->dir & FLAG_BIA)
        SAFE(fill_bia(p, bia_dt, bia_fp, r), WARN);
    if (p->dir & FLAG_FWD) {
        mkldnn_primitive_at_t inputs[3] = { {src_dt.p_, 0}, {wei_dt.p_, 0},
            {p->dir & FLAG_BIA ? bia_dt.p_ : NULL, 0}
        };
        const_mkldnn_primitive_t outputs[] = { dst_dt.p_ };
        DNN_SAFE(mkldnn_primitive_create(&c, dpd, inputs, outputs), WARN);
        SAFE(execute(c), WARN);
        if (bench_mode & CORR) {
            compute_ref_bwd_d(&p_tr, dst_fp, wei_tr_fp, bia_fp, src_fp);
            dnn_mem_t dst(dst_dt, fp, src_format);
            SAFE(dst.reorder(dst_dt), WARN);
            SAFE(compare_dst(p, dst, dst_fp, r, true), WARN);
        }
    } else if (p->dir == BWD_D) {
        mkldnn_primitive_at_t inputs[3] = { {dst_dt.p_, 0}, {wei_dt.p_, 0}, };
        const_mkldnn_primitive_t outputs[] = { src_dt.p_ };
        DNN_SAFE(mkldnn_primitive_create(&c, dpd, inputs, outputs), WARN);
        SAFE(execute(c), WARN);
        if (bench_mode & CORR) {
            compute_ref_fwd(&p_tr, dst_fp, wei_tr_fp, zero_fp, src_fp);
            dnn_mem_t src(src_dt, fp, src_format);
            SAFE(src.reorder(src_dt), WARN);
            SAFE(compare_src(p, src, src_fp, r, true), WARN);
        }
    } else if (p->dir & FLAG_BWD && p->dir & FLAG_WEI) {
        mkldnn_primitive_at_t inputs[3] = { {src_dt.p_, 0}, {dst_dt.p_, 0}, };
        const_mkldnn_primitive_t outputs[] = { wei_dt.p_,
            p->dir & FLAG_BIA ? bia_dt.p_ : NULL,
        };
        DNN_SAFE(mkldnn_primitive_create(&c, dpd, inputs, outputs), WARN);
        SAFE(execute(c), WARN);
        if (bench_mode & CORR) {
            compute_ref_bwd_weights(&p_tr, dst_fp, wei_tr_fp, src_fp);
            transpose_data_wei(&p_tr, wei_tr_fp, wei_fp);
            dnn_mem_t wei(wei_dt, fp, wei_format);
            SAFE(wei.reorder(wei_dt), WARN);
            SAFE(compare_wei(&p_tr, wei, wei_fp, r, true), WARN);
            if (p->dir & FLAG_BIA) {
                compute_ref_bwd_bias(p, bia_fp, dst_fp);
                dnn_mem_t bia(bia_dt, fp, mkldnn_x);
                SAFE(bia.reorder(bia_dt), WARN);
                SAFE(compare_bia(p, bia, bia_fp, r, true), WARN);
            }
        }
    } else {
        delete p_bia_dt;
        delete p_bia_fp;
        delete p_zero_fp;
        SAFE(FAIL, CRIT);
    }

    if (bench_mode & PERF) {
        auto &t = r->timer;
        t.reset();
        while (true) {
            SAFE(execute(c), WARN);
            t.stamp();
            const bool stop = false
                || (fix_times_per_prb && t.times() >= fix_times_per_prb)
                || (!fix_times_per_prb
                        && t.total_ms() >= max_ms_per_prb
                        && t.times() >= min_times_per_prb);
            if (stop) break;
        }
    }

    DNN_SAFE_V(mkldnn_primitive_destroy(c));
    DNN_SAFE_V(mkldnn_primitive_desc_destroy(dpd));

    delete p_bia_dt;
    delete p_bia_fp;
    delete p_zero_fp;

   return OK;
}

}
