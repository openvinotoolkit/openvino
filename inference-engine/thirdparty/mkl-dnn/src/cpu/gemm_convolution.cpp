/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include <common/primitive_attr.hpp>
#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "gemm_convolution.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"

#include "ref_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <bool with_relu>
void _gemm_convolution_fwd_t<with_relu>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t*>(this->memory());

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;
    const int MB = conf_.MB();

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());

    const int M = jcp.os * jcp.od;
    const size_t src_step = (src_d.blk_off(1) - src_d.off_l(0)) / jcp.ngroups;
    const size_t dst_step = (dst_d.blk_off(1) - dst_d.off_l(0)) / jcp.ngroups;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;
    src += src_d.off_l(0);
    dst += dst_d.off_l(0);

    const int K = jcp.ic * jcp.ks;
    const int N = jcp.oc;
    const int m = jcp.os;
    const int LDA = jcp.im2col_sz ? m : M;

    const data_t one = 1.0;

    data_t *col = (jcp.im2col_sz)
        ? (data_t *)this->scratchpad_->get()
        : nullptr;

    parallel_nd(jcp.im2col_sz * jcp.nthr,
            [&](ptrdiff_t i) { col[i] = (data_t)0; });

    const size_t work_amount = jcp.ngroups * MB * jcp.od;
    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

        int g{0}, n{0}, od{0};
        size_t start = 0, end = 0;

        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(start, g, jcp.ngroups, n, MB, od, jcp.od);

        for (size_t iwork = start; iwork < end; ++iwork) {
            const data_t *_src = src + (n * jcp.ngroups + g) * src_step;
            const data_t *_weights = weights + g * weights_g_size;
            data_t *_dst = dst + (n * jcp.ngroups + g) * dst_step;

            if (jcp.im2col_sz) {
                if (jcp.id == 1)
                    jit_gemm_convolution_utils::im2col(jcp, _src, _col);
                else
                    jit_gemm_convolution_utils::im2col_3d(jcp, _src, _col, od);
            }

            const data_t one = 1.0;
            extended_sgemm("N", "N", &m, &N, &K, &one,
                    jcp.im2col_sz ? _col : _src + od * m, &LDA, _weights, &K,
                    &this->beta_, _dst + od * m, &M);

            const auto &p = conf_.attr()->post_ops_;
            bool need_bias = jcp.with_bias;
            if (use_fast_relu) {
                data_t *d = _dst + od * m;

                for (int oc = 0; oc < jcp.oc; ++oc) {
                    data_t b = need_bias ? bias[g * jcp.oc + oc] : 0;
                    for (int oS = 0; oS < m; ++oS) {
                        d[oS] += b;
                        if (d[oS] < 0) d[oS] *= fast_relu_ns;
                    }
                    d += M;
                }

                need_bias = false;
            } else if (p.len_ > 0) {
                int eltwise_inj_idx = 0;
                int depthwise_inj_idx = 0;

                for (int i = 0; i < p.len_; i++) {
                    data_t *d = _dst + od * m;
                    auto& post_op = p.entry_[i];
                    if (post_op.is_eltwise()) {
                        for (int oc = 0; oc < jcp.oc; ++oc) {
                            data_t b = need_bias ? bias[g * jcp.oc + oc] : 0;
                            for (int oS = 0; oS < m; ++oS) {
                                d[oS] += b;
                                d[oS] = eltwise_injectors[eltwise_inj_idx]->compute_scalar(d[oS]);
                            }
                            d += M;
                        }

                        eltwise_inj_idx++;
                        need_bias = false;
                    } else if (post_op.is_depthwise()) {
                        auto depthwise_weights = post_op.depthwise.weights_data;
                        auto depthwise_bias = post_op.depthwise.biases_data;

                        for (int oc = 0; oc < jcp.oc; ++oc) {
                            data_t b = need_bias ? bias[g * jcp.oc + oc] : 0;
                            for (int oS = 0; oS < m; ++oS) {
                                d[oS] += b;
                                d[oS] = depthwise_injectors[depthwise_inj_idx]->compute_scalar(d[oS],
                                                                  depthwise_weights + g * jcp.oc + oc,
                                                                  depthwise_bias + g * jcp.oc + oc);
                            }
                            d += M;
                        }

                        depthwise_inj_idx++;
                        need_bias = false;
                    }
                }
            }

            if (need_bias) {
                data_t *d = _dst + od * m;

                for (int oc = 0; oc < jcp.oc; ++oc) {
                    data_t b = bias[g * jcp.oc + oc];
                    for (int oS = 0; oS < m; ++oS) {
                        d[oS] += b;
                    }
                    d += M;
                }
            }

            nd_iterator_step(g, jcp.ngroups, n, MB, od, jcp.od);
        }
    });
}

void gemm_convolution_bwd_data_t::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory());

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;
    const int MB = conf_.MB();

    const int M = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * M;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int m = jcp.os;
    const int K = jcp.oc;
    const int N = jcp.ic * jcp.ks;
    const int LDC = jcp.im2col_sz ? m : M;
    data_t *col = jcp.im2col_sz ? (data_t *)this->scratchpad_->get() : nullptr;

    parallel_nd(jcp.im2col_sz * jcp.nthr,
            [&](ptrdiff_t i) { col[i] = (data_t)0; });

    const size_t work_amount = (size_t)jcp.ngroups * MB;

    if (jcp.id > 1) {
        const ptrdiff_t diff_src_sz = (ptrdiff_t)(work_amount * src_step);
        parallel_nd(diff_src_sz, [&](ptrdiff_t i) { diff_src[i] = (data_t)0; });
    }

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

        int g{0}, n{0};
        size_t start = 0, end = 0;
        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(start, g, jcp.ngroups, n, MB);
        for (size_t iwork = start; iwork < end; ++iwork) {

            data_t *_diff_src = diff_src + (n * jcp.ngroups + g)*src_step;
            const data_t *_weights = weights + g * weights_g_size;
            for (int od = 0; od < jcp.od; ++od) {
                const data_t *_diff_dst = diff_dst + (n * jcp.ngroups + g)
                    *dst_step + od * m;

                const data_t zero = 0.0, one = 1.0;
                extended_sgemm("N", "T", &m, &N, &K, &one, _diff_dst, &M,
                    _weights, &N, &zero,
                    jcp.im2col_sz ? _col:_diff_src + od * m, &LDC);

                if (jcp.im2col_sz) {
                    if (jcp.id == 1)
                        jit_gemm_convolution_utils::col2im(jcp, _col,
                            _diff_src);
                    else
                        jit_gemm_convolution_utils::col2im_3d(jcp, _col,
                            _diff_src, od);
                }
            }
            nd_iterator_step(g, jcp.ngroups, n, MB);
        }
    });
}

void gemm_convolution_bwd_weights_t::execute_backward_weights() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t*>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;
    const int K = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * K;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int k = jcp.os;
    const int N = jcp.oc;
    const int M = jcp.ic * jcp.ks;
    const int LDA = jcp.im2col_sz ? k : K;

    data_t *col = nullptr, *wei_reduction = nullptr;
    ptrdiff_t wei_offset = 0;
    if (jcp.im2col_sz) {
        col = (data_t *)this->scratchpad_->get();
        wei_offset = jcp.im2col_sz * jcp.nthr;
    }
    if (jcp.need_wei_reduction)
        wei_reduction = (data_t *)this->scratchpad_->get() + wei_offset;

    parallel_nd(jcp.im2col_sz * jcp.nthr,
            [&](ptrdiff_t i) { col[i] = (data_t)0; });

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int ithr_g, nthr_g, ithr_mb, nthr_mb;
        size_t g_start{0}, g_end{0}, mb_start{0}, mb_end{0};

        const int mb_for_balance = jcp.need_wei_reduction ? jcp.mb : 1;
        jit_gemm_convolution_utils::bwd_weights_balance(ithr, nthr, jcp.ngroups,
                mb_for_balance, ithr_g, nthr_g, ithr_mb, nthr_mb);

        assert(utils::implication(!jcp.need_wei_reduction, nthr_mb == 1));
        const int need_reduction = nthr_mb != 1;

        if (ithr_g != -1 && ithr_mb != -1) {
            balance211((size_t)jcp.ngroups, nthr_g, ithr_g, g_start, g_end);
            balance211((size_t)jcp.mb, nthr_mb, ithr_mb, mb_start, mb_end);

            assert(implication((g_end - g_start) > 1, need_reduction == 0));

            data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;
            data_t *weights_reduce_base = wei_reduction
                    + ithr_g * nthr_mb * weights_g_size;
            data_t *weights_reduce = weights_reduce_base
                    + ithr_mb * weights_g_size;

            for (size_t g = g_start; g < g_end; ++g) {
                data_t *_diff_weights = need_reduction
                        ? weights_reduce : (diff_weights + g * weights_g_size);
                for (size_t mb = mb_start; mb < mb_end; ++mb) {
                    const data_t *_src = src + (mb*jcp.ngroups+g)*src_step;
                    for (int od = 0; od < jcp.od; ++od) {
                    const data_t *_diff_dst = diff_dst
                            + (mb*jcp.ngroups+g)*dst_step + od * k;

                    if (jcp.im2col_sz) {
                        if (jcp.id == 1)
                            jit_gemm_convolution_utils::im2col(jcp, _src, _col);
                        else
                            jit_gemm_convolution_utils::im2col_3d(jcp, _src,
                                _col, od);
                    }

                    const data_t zero = 0.0, one = 1.0;
                    extended_sgemm(
                        "T", "N", &M, &N, &k, &one,
                        jcp.im2col_sz ? _col : _src + od * k,
                        &LDA, _diff_dst, &K,
                        mb == mb_start && od == 0 ? &zero : &one,
                        _diff_weights, &M);
                    }
                }
            }
            if (need_reduction) {
                mkldnn_thr_barrier();
                data_t *weights_base = diff_weights + g_start * weights_g_size;
                jit_gemm_convolution_utils::bwd_weights_reduction_par(
                    ithr_mb, nthr_mb, jcp, weights_reduce_base, weights_base);
            }
        } else
            if (need_reduction) { mkldnn_thr_barrier(); }
    });

    if (jcp.with_bias) {
        parallel_nd(jcp.ngroups, jcp.oc, [&](int g, int oc) {
            data_t db = 0;
            size_t offset_ = (size_t)g * dst_step + (size_t)oc * K;
            for (int mb = 0; mb < jcp.mb; ++mb)
            {
                size_t offset = offset_ + (size_t)mb * jcp.ngroups * dst_step;
                for (int od = 0; od < jcp.od; ++od)
                for (int oh = 0; oh < jcp.oh; ++oh)
                PRAGMA_OMP_SIMD(reduction(+:db))
                for (int ow = 0; ow < jcp.ow; ++ow) {
                    db += diff_dst[offset];
                    offset++;
                }
            }
            diff_bias[g*jcp.oc+oc] = db;
            nd_iterator_step(g, jcp.ngroups, oc, jcp.oc);
        });
    }
}

template struct _gemm_convolution_fwd_t<true>;
template struct _gemm_convolution_fwd_t<false>;
}
}
}
