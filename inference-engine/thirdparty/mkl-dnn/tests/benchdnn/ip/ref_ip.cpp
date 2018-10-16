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

#include "ip/ip.hpp"

namespace ip {

void compute_ref_fwd(const prb_t *p, dnn_mem_t &src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m, dnn_mem_t &dst_m) {
    auto ker = [&](float &d, int mb, int oc) {
        for (int ic = 0; ic < p->ic; ++ic) {
            for (int ih = 0; ih < p->ih; ++ih) {
            for (int iw = 0; iw < p->iw; ++iw) {
                size_t src_off = src_off_f(p, mb, ic, ih, iw);
                size_t wei_off = wei_off_f(p, oc, ic, ih, iw);
                d += ((float*)src_m)[src_off] * ((float*)wei_m)[wei_off];
            }
            }
        }
    };

#   pragma omp parallel for collapse(2)
    for (int mb = 0; mb < p->mb; ++mb) {
        for (int oc = 0; oc < p->oc; ++oc) {
            size_t dst_off = dst_off_f(p, mb, oc);
            size_t bia_off = bia_off_f(p, oc);
            float &d = ((float*)dst_m)[dst_off];
            d = (p->dir & FLAG_BIA) ? ((float*)bia_m)[bia_off] : 0;
            ker(d, mb, oc);
        }
    }
}

#if 0
void compute_ref_convolution_bwd_d(const prb_t *p, dnn_mem_t &diff_src_m,
        dnn_mem_t &wei_m, dnn_mem_t &diff_dst_m) {
    auto ker = [=](float &ds, int g, int mb, int ic, int ih, int iw) {
        for (int oc = 0; oc < p->oc/p->g; ++oc) {
            for (int kh = 0; kh < p->kh; ++kh) {
                int oh = ih - kh + p->ph;
                if (oh < 0 || oh % p->sh) continue;
                oh /= p->sh;
                if (oh >= p->oh) continue;

                for (int kw = 0; kw < p->kw; ++kw) {
                    int ow = iw - kw + p->pw;
                    if (ow < 0 || ow % p->sw) continue;
                    ow /= p->sw;
                    if (ow >= p->ow) continue;

                    size_t dst_off = dst_off_f(p, mb, g, oc, oh, ow);
                    size_t wei_off = wei_off_f(p, g, oc, ic, kh, kw);
                    ds += ((float*)diff_dst_m)[dst_off]
                        * ((float*)wei_m)[wei_off];
                }
            }
        }
    };

#   pragma omp parallel for collapse(5)
    for (int g = 0; g < p->g; ++g) {
    for (int mb = 0; mb < p->mb; ++mb) {
        for (int ic = 0; ic < p->ic/p->g; ++ic) {
        for (int ih = 0; ih < p->ih; ++ih) {
        for (int iw = 0; iw < p->iw; ++iw) {
            size_t src_off = src_off_f(p, mb, g, ic, ih, iw);
            float &ds = ((float*)diff_src_m)[src_off];
            ds = 0;
            ker(ds, g, mb, ic, ih, iw);
        }
        }
        }
    }
    }
}

void compute_ref_convolution_bwd_w(const prb_t *p, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m) {
    auto ker = [=](float &dw, int g, int oc, int ic, int kh, int kw) {
        for (int mb = 0; mb < p->mb; ++mb) {
            for (int oh = 0; oh < p->oh; ++oh) {
            for (int ow = 0; ow < p->ow; ++ow) {
                const int ih = oh * p->sh - p->ph + kh;
                const int iw = ow * p->sw - p->pw + kw;
                if (ih < 0 || ih >= p->ih) continue;
                if (iw < 0 || iw >= p->iw) continue;

                size_t src_off = src_off_f(p, mb, g, ic, ih, iw);
                size_t dst_off = dst_off_f(p, mb, g, oc, oh, ow);
                dw += ((float*)diff_dst_m)[dst_off]
                    * ((float*)src_m)[src_off];
            }
            }
        }
    };

#   pragma omp parallel for collapse(5)
    for (int g = 0; g < p->g; ++g) {
        for (int oc = 0; oc < p->oc/p->g; ++oc) {
        for (int ic = 0; ic < p->ic/p->g; ++ic) {
            for (int kh = 0; kh < p->kh; ++kh) {
            for (int kw = 0; kw < p->kw; ++kw) {
                size_t wei_off = wei_off_f(p, g, oc, ic, kh, kw);
                float &dw = ((float*)diff_wei_m)[wei_off];
                dw = 0;
                ker(dw, g, oc, ic, kh, kw);
            }
            }
        }
        }
    }

    if (!(p->dir & FLAG_BIA)) return;

#   pragma omp parallel for collapse(2)
    for (int g = 0; g < p->g; ++g) {
        for (int oc = 0; oc < p->oc/p->g; ++oc) {
            size_t bia_off = bia_off_f(p, g, oc);
            float &db = ((float*)diff_bia_m)[bia_off];
            db = 0;

            for (int mb = 0; mb < p->mb; ++mb) {
                for (int oh = 0; oh < p->oh; ++oh) {
                for (int ow = 0; ow < p->ow; ++ow) {
                    size_t dst_off = dst_off_f(p, mb, g, oc, oh, ow);
                    db += ((float*)diff_dst_m)[dst_off];
                }
                }
            }
        }
    }
}
#endif

}
