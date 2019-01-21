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

#include "src/common/mkldnn_thread.hpp"

#include "conv/conv_common.hpp"

namespace conv {

void compute_ref_fwd(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &dst_m) {
    if (p->alg == WINO && p->cfg[SRC].dt == mkldnn_f32) {
        compute_wino_ref_fwd(p, src_m, wei_m, bia_m, dst_m);
    } else {
        compute_ref_direct_fwd(p, src_m, wei_m, bia_m, dst_m);
    }
}

void compute_ref_bwd_d(const prb_t *p, dnn_mem_t &diff_src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &diff_dst_m) {
    if (p->alg == WINO && p->cfg[SRC].dt == mkldnn_f32) {
        compute_wino_ref_bwd_d(p, diff_src_m, wei_m, bia_m, diff_dst_m);
    } else {
        compute_ref_direct_bwd_d(p, diff_src_m, wei_m, bia_m, diff_dst_m);
    }
}

void compute_ref_bwd_w(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &diff_wei_m,
        dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m) {
    if (p->alg == WINO && p->cfg[SRC].dt == mkldnn_f32) {
        compute_wino_ref_bwd_w(p, src_m, diff_wei_m, diff_bia_m, diff_dst_m);
    } else {
        compute_ref_direct_bwd_w(p, src_m, diff_wei_m, diff_bia_m, diff_dst_m);
    }
}

void compute_ref_direct_fwd(const prb_t *p, dnn_mem_t &src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m, dnn_mem_t &dst_m) {
    auto ker = [&](float &d, int g, int mb, int oc, int od, int oh, int ow) {
        for (int ic = 0; ic < p->ic/p->g; ++ic) {
            for (int kd = 0; kd < p->kd; ++kd) {
                const int id = od * p->sd - p->pd + kd * (p->dd + 1);
                if (id < 0 || id >= p->id) continue;
                for (int kh = 0; kh < p->kh; ++kh) {
                    const int ih = oh * p->sh - p->ph + kh * (p->dh + 1);
                    if (ih < 0 || ih >= p->ih) continue;

                    for (int kw = 0; kw < p->kw; ++kw) {
                        const int iw = ow * p->sw - p->pw + kw * (p->dw + 1);
                        if (iw < 0 || iw >= p->iw) continue;

                        size_t src_off = src_off_f(p, mb, g, ic, id, ih, iw);
                        size_t wei_off = wei_off_f(p, g, oc, ic, kd, kh, kw);
                        d += ((float*)src_m)[src_off]
                            * ((float*)wei_m)[wei_off];
                    }
                }
            }
        }
    };

    auto maybe_scale = [&](float &d, int oc) {
        if (!p->attr.oscale.is_def()) {
            using policy_t = attr_t::scale_t::policy_t;
            const auto &s = p->attr.oscale;
            if (s.policy == policy_t::COMMON) {
                d *= s.scale;
            } else {
                d *= p->scales[oc];
            }
        }
    };

    auto maybe_post_ops = [&](float &conv_res, float dst) {
        const auto &ops = p->attr.post_ops;
        for (int idx = 0; idx < ops.len; ++idx) {
            using pk = attr_t::post_ops_t::kind_t;
            const auto &e = ops.entry[idx];
            switch (e.kind) {
            case pk::SUM:
                conv_res += e.sum.scale * dst;
                break;
            case pk::RELU:
                conv_res = e.eltwise.scale * (conv_res < 0 ? 0 : conv_res);
                break;
            default:
                assert(!"unknown attr::post_ops::kind");
            }
        }
    };

    mkldnn::impl::parallel_nd(p->g, p->mb, p->oc / p->g, p->od, p->oh, p->ow,
        [&](int g, int mb, int oc, int od, int oh, int ow) {
            const size_t dst_off = dst_off_f(p, mb, g, oc, od, oh, ow);
            float &dst = ((float*)dst_m)[dst_off];

            float conv_res = 0;
            ker(conv_res, g, mb, oc, od, oh, ow);

            if (p->dir & FLAG_BIA) {
                const size_t bia_off = bia_off_f(p, g, oc);
                conv_res += ((float*)bia_m)[bia_off];
            }

            if (p->merge == RELU && conv_res < 0)
                conv_res = 0;

            maybe_scale(conv_res, g * p->oc / p->g + oc);
            maybe_post_ops(conv_res, dst);

            dst = conv_res;
        }
    );
}

void compute_ref_direct_bwd_d(const prb_t *p, dnn_mem_t &diff_src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m, dnn_mem_t &diff_dst_m) {
    enum { precompute_size = 16 };
    const bool fast = MAX2(p->kh, p->kw) <= precompute_size;

    /* pre-computes arrays of oh(ow) and kh(kw) for traversing in kernel */
    auto precompute_ok = [](int i, int O, int K, int S, int P, int D,
            int &num, int *_o, int *_k) {
        assert(K <= precompute_size);
        num = 0;
        for (int k = 0; k < K; ++k) {
            int o = i - k * (D + 1) + P;
            if (o < 0 || o % S) continue;
            o /= S;
            if (o >= O) continue;
            _k[num] = k;
            _o[num] = o;
            ++num;
        }
    };

    auto ker_fast = [&](float &ds, int g, int mb, int ic, int id, int ih, int iw) {
        int kd[precompute_size], od[precompute_size], num_d;
        int kh[precompute_size], oh[precompute_size], num_h;
        int kw[precompute_size], ow[precompute_size], num_w;
        precompute_ok(id, p->od, p->kd, p->sd, p->pd, p->dd, num_d, od, kd);
        precompute_ok(ih, p->oh, p->kh, p->sh, p->ph, p->dh, num_h, oh, kh);
        precompute_ok(iw, p->ow, p->kw, p->sw, p->pw, p->dw, num_w, ow, kw);

        for (int oc = 0; oc < p->oc/p->g; ++oc) {
            for (int d = 0; d < num_d; ++d) {
                for (int h = 0; h < num_h; ++h) {
                    for (int w = 0; w < num_w; ++w) {

                        size_t dst_off = dst_off_f(p, mb, g, oc, od[d], oh[h], ow[w]);
                        size_t wei_off = wei_off_f(p, g, oc, ic, kd[d], kh[h], kw[w]);
                        ds += ((float*)diff_dst_m)[dst_off]
                        * ((float*)wei_m)[wei_off];
                    }
                }
            }
        }
    };

    auto ker = [&](float &ds, int g, int mb, int ic, int id, int ih, int iw) {
        for (int oc = 0; oc < p->oc/p->g; ++oc) {
            for (int kd = 0; kd < p->kd; ++kd) {
                int od = id - kd * (p->dd + 1) + p->pd;
                if (od < 0 || od % p->sd) continue;
                od /= p->sd;
                if (od >= p->od) continue;
                for (int kh = 0; kh < p->kh; ++kh) {
                    int oh = ih - kh * (p->dh + 1) + p->ph;
                    if (oh < 0 || oh % p->sh) continue;
                    oh /= p->sh;
                    if (oh >= p->oh) continue;

                    for (int kw = 0; kw < p->kw; ++kw) {
                        int ow = iw - kw * (p->dw + 1) + p->pw;
                        if (ow < 0 || ow % p->sw) continue;
                        ow /= p->sw;
                        if (ow >= p->ow) continue;

                        size_t dst_off = dst_off_f(p, mb, g, oc, od, oh, ow);
                        size_t wei_off = wei_off_f(p, g, oc, ic, kd, kh, kw);
                        ds += ((float*)diff_dst_m)[dst_off]
                        * ((float*)wei_m)[wei_off];
                    }
                }
            }
        }
    };

    auto maybe_scale = [&](float &ds, int ic) {
        if (!p->attr.oscale.is_def()) {
            using policy_t = attr_t::scale_t::policy_t;
            const auto &s = p->attr.oscale;
            if (s.policy == policy_t::COMMON) {
                ds *= s.scale;
            } else {
                ds *= p->scales[ic];
            }
        }
    };

    mkldnn::impl::parallel_nd(p->g, p->mb, p->ic / p->g, p->id, p->ih, p->iw,
        [&](int g, int mb, int ic, int id, int ih, int iw) {
            size_t src_off = src_off_f(p, mb, g, ic, id, ih, iw);
            float &ds = ((float*)diff_src_m)[src_off];
            ds = 0;
            if (fast)
                ker_fast(ds, g, mb, ic, id, ih, iw);
            else
                ker(ds, g, mb, ic, id, ih, iw);

            if (p->dir & FLAG_BIA) {
                const size_t bia_off = (size_t)g * p->ic / p->g + ic;
                ds += ((float*)bia_m)[bia_off];
            }
            maybe_scale(ds, g * p->ic / p->g + ic);
        }
    );
}

void compute_ref_bwd_weights(const prb_t *p, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_dst_m) {
    auto compute_bounds = [](int I, int O, int k, int S, int P, int D,
            int &o_s, int &o_e) {
        const float tmp = P - k * (D + 1);
        o_s = MAX2(0, ceilf(tmp / S));
        o_e = MIN2(O, ceilf((I + tmp) / S));
    };

    auto ker = [&](float &dw, int g, int oc, int ic, int kd, int kh, int kw) {
        int od_s, od_e, oh_s, oh_e, ow_s, ow_e;
        compute_bounds(p->id, p->od, kd, p->sd, p->pd, p->dd, od_s, od_e);
        compute_bounds(p->ih, p->oh, kh, p->sh, p->ph, p->dh, oh_s, oh_e);
        compute_bounds(p->iw, p->ow, kw, p->sw, p->pw, p->dw, ow_s, ow_e);

        for (int mb = 0; mb < p->mb; ++mb) {
            for (int od = od_s; od < od_e; ++od) {
            for (int oh = oh_s; oh < oh_e; ++oh) {
            for (int ow = ow_s; ow < ow_e; ++ow) {
                const int id = od * p->sd - p->pd + kd * (p->dd + 1);
                const int ih = oh * p->sh - p->ph + kh * (p->dh + 1);
                const int iw = ow * p->sw - p->pw + kw * (p->dw + 1);

                size_t src_off = src_off_f(p, mb, g, ic, id, ih, iw);
                size_t dst_off = dst_off_f(p, mb, g, oc, od, oh, ow);
                dw += ((float*)diff_dst_m)[dst_off]
                    * ((float*)src_m)[src_off];
            }
            }
            }
        }
    };

    mkldnn::impl::parallel_nd(
        p->g, p->oc / p->g, p->ic / p->g, p->kd, p->kh, p->kw,
        [&](int g, int oc, int ic, int kd, int kh, int kw) {
                size_t wei_off = wei_off_f(p, g, oc, ic, kd, kh, kw);
                float &dw = ((float*)diff_wei_m)[wei_off];
                dw = 0;
                ker(dw, g, oc, ic, kd, kh, kw);
        }
    );
}

void compute_ref_bwd_bias(const prb_t *p, dnn_mem_t &diff_bia_m,
    dnn_mem_t &diff_dst_m) {
    mkldnn::impl::parallel_nd(p->g, p->oc / p->g, [&](int g, int oc) {
       size_t bia_off = bia_off_f(p, g, oc);
       double sum = 0;

       for (int mb = 0; mb < p->mb; ++mb)
       for (int od = 0; od < p->od; ++od)
       for (int oh = 0; oh < p->oh; ++oh)
       for (int ow = 0; ow < p->ow; ++ow)
       {
           size_t dst_off = dst_off_f(p, mb, g, oc, od, oh, ow);
           sum += ((float*)diff_dst_m)[dst_off];
       }
       ((float *)diff_bia_m)[bia_off] = (float)sum;
    });
}

void compute_ref_direct_bwd_w(const prb_t *p, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m) {
    compute_ref_bwd_weights(p, src_m, diff_wei_m, diff_dst_m);
    if (!(p->dir & FLAG_BIA)) return;
    compute_ref_bwd_bias(p, diff_bia_m, diff_dst_m);
}

}
