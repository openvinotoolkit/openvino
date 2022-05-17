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

#include "jit_uni_planar_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

#define src_blk_off(f, n, c, d, h, w) \
    pd()->ndims() == 5 \
        ? (f).blk_off(n, c, d, h, w) \
        : (f).blk_off(n, c, h, w)

#define wht_blk_off(f, g, oc, ic, kd, kh, kw) \
    pd()->ndims() == 5 \
        ? pd()->with_groups() \
            ? (f).blk_off(g, oc, ic, kd, kh, kw) \
            : (f).blk_off(oc, ic, kd, kh, kw) \
        : pd()->with_groups() \
            ? (f).blk_off(g, oc, ic, kh, kw) \
            : (f).blk_off(oc, ic, kh, kw)

template <cpu_isa_t isa>
void _jit_uni_planar_convolution_fwd_t<isa>::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto &jcp = pd()->jcp_;

    std::vector<int> od_indexes(jcp.od);

    int idx = 0;
    for (int i = 0; i < (jcp.dilate_d + 1); i++) {
        for (int ib = 0; ib < jcp.od; ib += (jcp.dilate_d + 1)) {
            if (ib + i >= jcp.od)
                continue;

            od_indexes[idx++] = ib + i;
            if (idx >= jcp.od)
                break;
        }
        if (idx >= jcp.od)
            break;
    }

    int threads_count = dnnl_get_max_threads();
    int odb_size = div_up(jcp.od, threads_count);

    auto kernel_params = [&](int n, int g, int icb, int oc, int od, int oh, int oh_blocks, int id, int wd, int kd_padding) {
        auto par_conv = jit_conv_call_s();

        const int hj = oh * jcp.stride_h;
        const int i_t_overflow = nstl::max(0, jcp.t_pad - hj);
        const int i_b_overflow = nstl::max(jcp.ih, hj + (jcp.kh - 1) * (jcp.dilate_h + 1) - jcp.t_pad + 1) - jcp.ih;
        const int ih = nstl::max(hj - jcp.t_pad + div_up(i_t_overflow, (jcp.dilate_h + 1)) * (jcp.dilate_h + 1), 0);
        const int wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
        const int kh_padding = jcp.kh - div_up(i_t_overflow, (jcp.dilate_h + 1)) - div_up(i_b_overflow, (jcp.dilate_h + 1));

        const size_t _oc = oc;
        const size_t _ic = g * jcp.nb_ic + icb;

        par_conv.src = &src[src_blk_off(src_d, n, _ic, id, ih, 0)];
        par_conv.dst = &dst[src_blk_off(dst_d, n, _oc, od, oh, 0)];
        par_conv.filt = &weights[wht_blk_off(weights_d, g, _oc, _ic, wd, wh, 0)];

        if (icb == 0) {
            if (bias)
                par_conv.bias = &bias[bias_d.blk_off(_oc)];
            par_conv.flags |= FLAG_IC_FIRST;
        }

        if (icb + 1 == jcp.nb_ic) {
            par_conv.flags |= FLAG_IC_LAST;
        }

        par_conv.oc_off = _oc * sizeof(float);
        par_conv.oh_blocks = (size_t)oh_blocks;

        par_conv.kh_padding = (size_t)nstl::max(0, kh_padding);
        par_conv.kd_padding = (size_t)nstl::max(0, kd_padding);

        return par_conv;
    };

    auto ker = [&](const int ithr, const int nthr) {
        int g = 0;
        int oc = 0;

        for (int n = 0; n < MB; n++) {
            int icbb = 0;
            while (icbb < jcp.nb_ic) {
                int icb_step = jcp.nb_ic_blocking;
                int icb_step_rem = jcp.nb_ic - icbb;
                if (icb_step_rem < jcp.nb_ic_blocking_max)
                    icb_step = icb_step_rem;

                for (int icb = icbb; icb < icbb + icb_step; ++icb) {
                    for (int ohb = 0; ohb < (jcp.dilate_h + 1); ohb++) {
                        for (int oh = ohb; oh < jcp.oh; oh += (jcp.dilate_h + 1)) {
                            int od_idx_off = ithr * odb_size;
                            for (int od_idx = 0; od_idx < odb_size; od_idx++) {
                                if ((od_idx_off + od_idx) >= jcp.od || od_indexes[od_idx_off + od_idx] >= jcp.od)
                                    continue;
                                int od = od_indexes[od_idx_off + od_idx];

                                const int dj = od * jcp.stride_d;
                                const int d_t_overflow = nstl::max(0, jcp.f_pad - dj);
                                const int d_b_overflow =
                                        nstl::max(jcp.id, dj + (jcp.kd - 1) * (jcp.dilate_d + 1) - jcp.f_pad + 1) -
                                        jcp.id;
                                const int id = nstl::max(dj - jcp.f_pad +
                                                         div_up(d_t_overflow, (jcp.dilate_d + 1)) * (jcp.dilate_d + 1),
                                                         0);
                                const int wd = div_up(d_t_overflow, (jcp.dilate_d + 1));
                                const int kd_padding = jcp.kd - div_up(d_t_overflow, (jcp.dilate_d + 1)) -
                                                       div_up(d_b_overflow, (jcp.dilate_d + 1));

                                jit_conv_call_s par_conv = kernel_params(n, g, icb, oc, od, oh, 1, id, wd, kd_padding);

                                (*kernel_)(&par_conv);
                            }
                        }
                    }
                }
                icbb += icb_step;
            }
        }
    };

    parallel(0, ker);
}


template struct _jit_uni_planar_convolution_fwd_t<avx512_common>;
template struct _jit_uni_planar_convolution_fwd_t<avx2>;

}
}
}
}
