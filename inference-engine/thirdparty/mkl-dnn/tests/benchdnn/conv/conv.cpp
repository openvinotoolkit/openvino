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

#include "src/common/mkldnn_thread.hpp"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "norm.hpp"

#include "conv/conv_common.hpp"

namespace conv {

inline bool is_conv_3d(const prb_t *p)
{
    return (p->id > 1) ? 1 : 0;
}

inline bool is_conv_1d(const prb_t *p)
{
    return (!is_conv_3d(p) && p->ih == 1 && p->kh == 1
                   && p->cfg[SRC].dt != mkldnn_s8 // temporary workaround until
                   && p->cfg[SRC].dt != mkldnn_u8) // int8 jit supports 1d
            ? 1 : 0;
}

double get_trust_nz_level(const prb_t *p, data_kind_t kind, bool final_compare)
{
    if (!final_compare)
        return p->cfg[kind].f_sparsity;

    auto count_relu = [&]() {
        const auto &po = p->attr.post_ops;
        int count = 0;
        for (int i = 0; i < po.len; ++i)
            count += po.entry[i].kind == attr_t::post_ops_t::kind_t::RELU;
        count = MAX2(count, p->merge == RELU ? 1 : 0);
        return count;
    };

    double trust = 0.3; /* why? */
    switch (kind) {
        case SRC:
            trust /= p->sd * p->sh * p->sw;
            break;
        case WEI:
            trust /= 1. * p->kd * p->kh * p->kw
                / MIN3(p->kd * p->kh * p->kw, p->id * p->ih * p->iw
                , p->od * p->oh * p->ow);
            break;
        case BIA:
            trust = 0.8 * p->cfg[DST].f_sparsity; /* why? */
            break;
        case DST:
            trust /= count_relu() == 0 ? 1 : 2;
            break;
    }

    return trust;
}

inline double get_eps(const prb_t *p, const data_kind_t kind) {
    if (p->alg & WINO && p->dir & FLAG_WEI) {
        /*This is an empirical equation derived by observing growth error
          with increasing 'k' dimension in gemm of winograd*/
        return p->cfg[kind].eps *
            (MAX2(1, pow(10, 0.4 * log10(0.125 * p->mb * p->oh * p->ow))));
    }
    return p->cfg[kind].eps;
}

inline void get_result(const prb_t *p, const data_kind_t kind, res_t *r,
        const diff_norm_t diff_norm) {
    bool wino_test = (p->alg & WINO)
        && (diff_norm.rel_diff(norm_t::L2) <= get_eps(p, kind));
    /* Ignoring elementwise errors for winograd,
       since large relative error in few elements(which are anyways close to zero)
       results in false positive failures*/
    if (wino_test) r->errors = 0;
    r->state = r->errors ? FAILED : r->state;
}

inline int compare_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    size_t nelems = mem_dt.nelems();

    const char *skind = data_kind2str(kind);

    int in = 0, below = 0, above = 0;
    int in_ok = 0, below_ok = 0, above_ok = 0;
    int non_zero = 0;

    diff_norm_t diff_norm;

    r->errors = 0;
    r->total = nelems;

    for (size_t i = 0; i < nelems; ++i) {
        const float dt = ((float*)mem_dt)[i];
        const float fp0 = ((float*)mem_fp)[i];

        float fp = fp0;
        if (p->cfg[kind].dt != mkldnn_f32) {
            using R = attr_t::round_mode_t;
            switch (p->attr.irmode) {
                case R::DOWN: fp = floorf(fp0); break;
                case R::NEAREST: fp = nearbyintf(fp0); break;
                default:
                    return UNTESTED;
            }
        }

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);

        bool ok = true;
        if (fp < p->cfg[kind].min) {
            diff_norm.update(p->cfg[kind].min, dt);
            ok = dt == p->cfg[kind].min;
            below += 1;
            below_ok += ok;
        } else if (fp > p->cfg[kind].max) {
            diff_norm.update(p->cfg[kind].max, dt);
            ok = dt == p->cfg[kind].max;
            above += 1;
            above_ok += ok;
        } else {
            diff_norm.update(fp, dt);
            ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= get_eps(p, kind);
            in += 1;
            in_ok += ok;
        }
        if (!ok) {
            r->errors++;
            if ((!(p->alg & WINO) && r->errors < 10) || verbose >=10) {
                int mb_or_g = 0, g_or_oc = 0, c = 0, d = 0, h = 0, w = 0;
                switch (kind) {
                case SRC: inv_src_off_f(p, i, mb_or_g, g_or_oc, c, d, h, w); break;
                case WEI: inv_wei_off_f(p, i, mb_or_g, g_or_oc, c, d, h, w); break;
                case BIA: inv_bia_off_f(p, i, mb_or_g, g_or_oc); break;
                case DST: inv_dst_off_f(p, i, mb_or_g, g_or_oc, c, d, h, w); break;
                }
                print(0, "[%4lu][%s%s][%d,%d,%d,%d,%d,%d] "
                        "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                        (unsigned long)i,
                        final_compare == false ? "REORDER " : "",
                        skind, mb_or_g, g_or_oc, c, d, h, w,
                        fp, fp0, dt, diff, rel_diff);
            }
        }

        /* for debug purposes only: dump the output */
        if (final_compare && verbose >= 50 && i < 30) {
            int mb_or_g = 0, g_or_oc = 0, c = 0, d = 0, h = 0, w = 0;
            switch (kind) {
            case SRC: inv_src_off_f(p, i, mb_or_g, g_or_oc, c, d, h, w); break;
            case WEI: inv_wei_off_f(p, i, mb_or_g, g_or_oc, c, d, h, w); break;
            case BIA: inv_bia_off_f(p, i, mb_or_g, g_or_oc); break;
            case DST: inv_dst_off_f(p, i, mb_or_g, g_or_oc, c, d, h, w); break;
            }

            print(0, "[%4lu][%s][%d,%d,%d,%d,%d,%d] fp:%8g fp0:%8g dt:%8g\n",
                    (unsigned long)i,
                    skind, mb_or_g, g_or_oc, c, d, h, w, fp, fp0, dt);
        }

        non_zero += fp != 0;
    }

    diff_norm.done();

    if (final_compare || r->errors) {
        const int vl = r->errors ? 0 : 2;
        print(vl, "@@@ [%s] %sdiff: l0(``%g``) "
                "l1:(%g,%g,%g,``%g``) "
                "l2:(%g,%g,%g,``%g``) "
                "l8:(%g,%g,%g,``%g``)\n",
                skind, final_compare ? "final: " : "",
                diff_norm.rel_diff(norm_t::L0),
                diff_norm.a_[norm_t::L1], diff_norm.b_[norm_t::L1],
                diff_norm.diff_[norm_t::L1], diff_norm.rel_diff(norm_t::L1),
                diff_norm.a_[norm_t::L2], diff_norm.b_[norm_t::L2],
                diff_norm.diff_[norm_t::L2], diff_norm.rel_diff(norm_t::L2),
                diff_norm.a_[norm_t::L8], diff_norm.b_[norm_t::L8],
                diff_norm.diff_[norm_t::L8], diff_norm.rel_diff(norm_t::L8));
    }

    const double trust_rg_level = 0.3;
    const double trust_nz_level = get_trust_nz_level(p, kind, final_compare);

    const double trust_rg = (double)in / r->total;
    const double trust_nz = (double)non_zero / r->total;

    const bool no_trust = true /* ...in the test ...at all */
        && final_compare
        && (trust_rg < trust_rg_level || trust_nz < trust_nz_level);

    const bool dump = verbose >= 20
        || (verbose >= 10 && (trust_rg < 1. || trust_nz < 1.));
    if (dump) {
        print(0, "@@@ [%s] %strust range:%.2f nz:%.2f "
                "(level range:%.2f nz:%.2f). "
                "in:%d (ok:%d) below:%d (ok:%d) above:%d (ok:%d) nz:%d "
                "total:%lu\n", skind, final_compare ? "final: " : "",
                trust_rg, trust_nz, trust_rg_level, trust_nz_level, in, in_ok,
                below, below_ok, above, above_ok, non_zero,
                (unsigned long)r->total);
    }

    if (no_trust) {
        r->state = MISTRUSTED;
        print(0, "@@@ [%s] test-bug: trust is too low. "
                "range:%.2f (?<%.2f) nz:%.2f (?<%.2f) (nz: %d total: %lu)\n",
                skind, trust_rg, trust_rg_level, trust_nz, trust_nz_level,
                non_zero, (unsigned long)r->total);
    }

    get_result(p, kind, r, diff_norm);

    if (final_compare && r->state == UNTESTED)
        r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int compare_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare)
{ return compare_dat(p, SRC, mem_dt, mem_fp, r, final_compare); }
int compare_wei(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare)
{ return compare_dat(p, WEI, mem_dt, mem_fp, r, final_compare); }
int compare_bia(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare)
{ return compare_dat(p, BIA, mem_dt, mem_fp, r, final_compare); }
int compare_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare)
{ return compare_dat(p, DST, mem_dt, mem_fp, r, final_compare); }

int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r) {
    const bool extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t *p_mem_00 = extra_mem
        ? new dnn_mem_t(mem_dt.md_, mkldnn_f32,
            get_default_format(mem_dt.md_.ndims, DATA))
        : &mem_fp;
    dnn_mem_t &mem_00 = *p_mem_00;

    const auto &c = p->cfg[SRC];
    const int range = c.f_max - c.f_min + 1;

    mkldnn::impl::parallel_nd(p->mb, p->ic, p->id, p->ih, p->iw,
        [&](int mb, int ic, int id, int ih, int iw) {
        const int gen = 5 * id + 17 * ih + 13 * iw + 13 * mb + 19 * ic + 1637;
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value =
            non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float*)mem_00)[src_off_f(p, mb, 0, ic, id, ih, iw)] = value;
    });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_src(p, mem_fp, mem_00, r), WARN);
        delete &mem_00;
    }

    return OK;
}

int fill_wei(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
    res_t *r) {
    const bool wino_s8 = p->alg == WINO && p->cfg[WEI].dt == mkldnn_s8;
    const bool s8_s8 = p->cfg[WEI].dt == mkldnn_s8 && p->cfg[SRC].dt == mkldnn_s8;
    const bool diff_data_type = mem_dt.dt() != mem_fp.dt();
    const bool check_reorder = diff_data_type && !wino_s8 && !s8_s8;

    dnn_mem_t *p_mem_00 = check_reorder
        ? new dnn_mem_t(mem_dt.md_, mkldnn_f32,
            get_default_format(mem_dt.md_.ndims, GWEI))
        : &mem_fp;
    dnn_mem_t &mem_00 = *p_mem_00;

    const auto &c = p->cfg[WEI];
    const int range = c.f_max - c.f_min + 1;

    mkldnn::impl::parallel_nd(
        p->g, p->oc / p->g, p->ic / p->g, p->kd, p->kh, p->kw,
        [&](int g, int oc, int ic, int kd, int kh, int kw) {
        const int gen = 5 * kd + 17 * kh + 13 * kw + 13 * oc + 19 * ic + 38;
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value =
            non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float*)mem_00)[wei_off_f(p, g, oc, ic, kd, kh, kw)] = value;
    });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (check_reorder) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_wei(p, mem_fp, mem_00, r), WARN);
        delete &mem_00;
    }

    return OK;
}

int fill_bia(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r) {
    const bool extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t *p_mem_00 = extra_mem
        ? new dnn_mem_t(mem_dt.md_, mkldnn_f32, mkldnn_x)
        : &mem_fp;
    dnn_mem_t &mem_00 = *p_mem_00;

    const auto &c = p->cfg[BIA];
    const int range = c.f_max - c.f_min + 1;

    const size_t sz = mem_00.nelems();
    for (size_t i = 0; i < sz; ++i) {
        const int gen = (int)(19 * i);
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value =
            non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float*)mem_00)[i] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_bia(p, mem_fp, mem_00, r), WARN);
        delete &mem_00;
    }

    return OK;
}

int fill_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r) {
    const bool extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t *p_mem_00 = extra_mem
        ? new dnn_mem_t(mem_dt.md_, mkldnn_f32,
            get_default_format(mem_dt.md_.ndims, DATA))
        : &mem_fp;
    dnn_mem_t &mem_00 = *p_mem_00;

    const auto &c = p->cfg[DST];
    const int range = c.f_max - c.f_min + 1;

    mkldnn::impl::parallel_nd(p->mb, p->oc, p->od, p->oh, p->ow,
        [&](int mb, int oc, int od, int oh, int ow) {
        const int gen = 7 * od + 19 * oh + 17 * ow + 13 * mb + 13 * oc + 223;
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value =
            non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float*)mem_00)[dst_off_f(p, mb, 0, oc, od, oh, ow)] = value;
    });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_dst(p, mem_fp, mem_00, r), WARN);
        delete &mem_00;
    }

    return OK;
}

inline int init_pd(const prb_t *p, mkldnn_convolution_desc_t &cd,
        mkldnn_primitive_desc_t &cpd, res_t *r) {
    mkldnn_memory_desc_t src_d, wei_d, bia_d, dst_d;

    int ndims = is_conv_3d(p) ? 5 : is_conv_1d(p) ? 3 : 4;
    mkldnn_dims_t src_dims = {p->mb, p->ic, p->ih, p->iw};
    mkldnn_dims_t src_1d_dims = {p->mb, p->ic, p->iw};
    mkldnn_dims_t src_3d_dims = {p->mb, p->ic, p->id, p->ih, p->iw};
    mkldnn_dims_t wei_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kh, p->kw};
    mkldnn_dims_t wei_1d_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kw};
    mkldnn_dims_t wei_3d_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kd, p->kh, p->kw};
    mkldnn_dims_t bia_dims = {p->oc};
    mkldnn_dims_t dst_dims = {p->mb, p->oc, p->oh, p->ow};
    mkldnn_dims_t dst_1d_dims = {p->mb, p->oc, p->ow};
    mkldnn_dims_t dst_3d_dims = {p->mb, p->oc, p->od, p->oh, p->ow};

    DNN_SAFE(mkldnn_memory_desc_init(&src_d, ndims,
        is_conv_3d(p) ? src_3d_dims : is_conv_1d(p) ? src_1d_dims : src_dims,
        p->cfg[SRC].dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&wei_d, ndims + 1,
        is_conv_3d(p) ? wei_3d_dims :  is_conv_1d(p) ? wei_1d_dims : wei_dims,
        p->cfg[WEI].dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&bia_d, 1, bia_dims, p->cfg[BIA].dt,
        mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&dst_d, ndims,
        is_conv_3d(p) ? dst_3d_dims : is_conv_1d(p) ? dst_1d_dims : dst_dims,
        p->cfg[DST].dt, mkldnn_any), WARN);
    int strides_nd[] = {p->sd, p->sh, p->sw};
    int dilates_nd[] = {p->dd, p->dh, p->dw};
    int padding_nd[] = {p->pd, p->ph, p->pw};

    auto bph = [&](int ih, int oh, int kh, int sh, int ph, int dh) {
        return (oh - 1) * sh - ih + ((kh - 1) * (dh + 1) + 1) - ph;
    };
    int padding_r_nd[] = {
        bph(p->id, p->od, p->kd, p->sd, p->pd, p->dd),
        bph(p->ih, p->oh, p->kh, p->sh, p->ph, p->dh),
        bph(p->iw, p->ow, p->kw, p->sw, p->pw, p->dw)};

    int *strides = strides_nd + (5 - ndims);
    int *dilates = dilates_nd + (5 - ndims);
    int *padding = padding_nd + (5 - ndims);
    int *padding_r = padding_r_nd + (5 - ndims);

    mkldnn_alg_kind_t alg = mkldnn_convolution_direct;
    if (p->alg == WINO) alg = mkldnn_convolution_winograd;

    switch (p->dir) {
    case FWD_D: case FWD_B: case FWD_I:
        DNN_SAFE(mkldnn_dilated_convolution_forward_desc_init(&cd,
                    p->dir == FWD_I
                        ? mkldnn_forward_inference
                        : mkldnn_forward_training,
                    alg, &src_d, &wei_d,
                    p->dir == FWD_B ? &bia_d : NULL, &dst_d,
                    strides, dilates, padding, padding_r,
                    mkldnn_padding_zero), WARN);
        break;
    case BWD_D:
        DNN_SAFE(mkldnn_dilated_convolution_backward_data_desc_init(&cd, alg,
                    &src_d, &wei_d, &dst_d, strides, dilates, padding, padding_r,
                    mkldnn_padding_zero), WARN);
        break;
    case BWD_W: case BWD_WB:
        DNN_SAFE(mkldnn_dilated_convolution_backward_weights_desc_init(&cd,
                    alg, &src_d, &wei_d, p->dir == BWD_W ? NULL : &bia_d, &dst_d,
                    strides, dilates, padding, padding_r,
                    mkldnn_padding_zero), WARN);
        break;
    default: DNN_SAFE(mkldnn_invalid_arguments, CRIT);
    }

    DNN_SAFE(cd.accum_data_type == p->cfg[ACC].dt
            ? mkldnn_success : mkldnn_unimplemented, CRIT);

    auto mkldnn_attr = create_mkldnn_attr(p->attr, p->oc, p->scales);

    mkldnn_status_t init_status = mkldnn_success;
    if (p->merge == RELU) {
        mkldnn_convolution_relu_desc_t crd;
        DNN_SAFE(mkldnn_convolution_relu_desc_init(&crd, &cd, 0), WARN);
        init_status = mkldnn_primitive_desc_create_v2(&cpd, &crd, mkldnn_attr,
                engine, NULL);
    } else {
        init_status = mkldnn_primitive_desc_create_v2(&cpd, &cd, mkldnn_attr,
                engine, NULL);
    }

    mkldnn_primitive_attr_destroy(mkldnn_attr);

    if (init_status == mkldnn_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(cpd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: mkldnn implementation: %s\n", impl_str);
        DNN_SAFE(mkldnn_primitive_desc_destroy(cpd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "mkldnn implementation: %s\n", impl_str);
    }

    auto q = [=](mkldnn_query_t query, int index = 0) {
        return *mkldnn_primitive_desc_query_memory_d(
                mkldnn_primitive_desc_query_pd(cpd, query, index));
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

    mkldnn_convolution_desc_t cd;
    mkldnn_primitive_desc_t cpd;
    mkldnn_primitive_t c{};

    SAFE(init_pd(p, cd, cpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
        return OK;

    auto &src_dt_d = p->dir == BWD_D ? cd.diff_src_desc : cd.src_desc;
    auto &wei_dt_d = p->dir & FLAG_WEI ? cd.diff_weights_desc : cd.weights_desc;
    auto &bia_dt_d = p->dir & FLAG_BWD ? cd.diff_bias_desc : cd.bias_desc;
    auto &dst_dt_d = p->dir & FLAG_BWD ? cd.diff_dst_desc: cd.dst_desc;

    dnn_mem_t src_dt(src_dt_d, p->cfg[SRC].dt);
    dnn_mem_t wei_dt(wei_dt_d, p->cfg[WEI].dt);
    dnn_mem_t dst_dt(dst_dt_d, p->cfg[DST].dt);
    dnn_mem_t *p_bia_dt = p->dir & FLAG_BIA
        ? new dnn_mem_t(bia_dt_d, p->cfg[BIA].dt) : new dnn_mem_t();
    dnn_mem_t &bia_dt = *p_bia_dt;

    auto src_format = get_default_format(src_dt.md_.ndims, DATA);
    auto wei_format = get_default_format(wei_dt.md_.ndims, GWEI);

    const auto fp = mkldnn_f32;
    dnn_mem_t src_fp(src_dt_d, fp, src_format);
    dnn_mem_t wei_fp(wei_dt_d, fp, wei_format);
    dnn_mem_t dst_fp(dst_dt_d, fp, src_format);
    dnn_mem_t *p_bia_fp = p->dir & FLAG_BIA
        ? new dnn_mem_t(bia_dt_d, fp, mkldnn_x) : new dnn_mem_t();
    dnn_mem_t &bia_fp = *p_bia_fp;

    SAFE(fill_src(p, src_dt, src_fp, r), WARN);
    SAFE(fill_wei(p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_dst(p, dst_dt, dst_fp, r), WARN);
    if (p->dir & FLAG_BIA)
        SAFE(fill_bia(p, bia_dt, bia_fp, r), WARN);

    if (p->dir & FLAG_FWD) {
        mkldnn_primitive_at_t inputs[3] = { {src_dt.p_, 0}, {wei_dt.p_, 0},
            {p->dir & FLAG_BIA ? bia_dt.p_ : NULL, 0}
        };
        const_mkldnn_primitive_t outputs[] = { dst_dt.p_ };
        DNN_SAFE(mkldnn_primitive_create(&c, cpd, inputs, outputs), WARN);
        SAFE(execute(c), WARN);
        if (bench_mode & CORR) {
            compute_ref_fwd(p, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, src_format);
            SAFE(dst.reorder(dst_dt), WARN);
            SAFE(compare_dst(p, dst, dst_fp, r, true), WARN);
        }
    } else if (p->dir == BWD_D) {
        mkldnn_primitive_at_t inputs[3] = { {dst_dt.p_, 0}, {wei_dt.p_, 0}, };
        const_mkldnn_primitive_t outputs[] = { src_dt.p_ };
        DNN_SAFE(mkldnn_primitive_create(&c, cpd, inputs, outputs), WARN);
        SAFE(execute(c), WARN);
        if (bench_mode & CORR) {
            compute_ref_bwd_d(p, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t src(src_dt, fp, src_format);
            SAFE(src.reorder(src_dt), WARN);
            SAFE(compare_src(p, src, src_fp, r, true), WARN);
        }
    } else if (p->dir & FLAG_BWD && p->dir & FLAG_WEI) {
        mkldnn_primitive_at_t inputs[3] = { {src_dt.p_, 0}, {dst_dt.p_, 0}, };
        const_mkldnn_primitive_t outputs[] = { wei_dt.p_,
            p->dir & FLAG_BIA ? bia_dt.p_ : NULL,
        };
        DNN_SAFE(mkldnn_primitive_create(&c, cpd, inputs, outputs), WARN);
        SAFE(execute(c), WARN);
        if (bench_mode & CORR) {
            compute_ref_bwd_w(p, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t wei(wei_dt, fp, wei_format);
            SAFE(wei.reorder(wei_dt), WARN);
            SAFE(compare_wei(p, wei, wei_fp, r, true), WARN);
            if (p->dir & FLAG_BIA) {
                dnn_mem_t bia(bia_dt, fp, mkldnn_x);
                SAFE(bia.reorder(bia_dt), WARN);
                SAFE(compare_bia(p, bia, bia_fp, r, true), WARN);
            }
        }
    } else {
        delete p_bia_dt;
        delete p_bia_fp;
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

    DNN_SAFE(mkldnn_primitive_desc_destroy(cpd), CRIT);
    DNN_SAFE(mkldnn_primitive_destroy(c), CRIT);

    delete p_bia_dt;
    delete p_bia_fp;

    return OK;
}

}
