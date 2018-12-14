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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_traits.hpp"
#include "math_utils.hpp"

#include "ref_deconvolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

typedef float data_t;

void ref_deconvolution_fwd_t::compute_fwd_bias() {
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());
    const memory_desc_wrapper dst_d(conf_.dst_pd());

    const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OD = conf_.OD();
    const int OC = conf_.OC() / G;
    const int ndims = conf_.desc()->src_desc.ndims;

    parallel_nd(MB, G, OC, OD, OH, OW,
        [&](int mb, int g, int oc, int od, int oh, int ow) {
            auto b = bias[g * OC + oc];
            if (ndims == 5)
                dst[dst_d.off(mb, g*OC + oc, od, oh, ow)] += b;
            else
                dst[dst_d.off(mb, g*OC + oc, oh, ow)] += b;
    });
}

void ref_deconvolution_fwd_t::compute_fwd_bias_ncdhw() {
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper dst_d(conf_.dst_pd());

    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int SP = conf_.OW()*conf_.OH()*conf_.OD();

    parallel_nd(MB, OC, [&](int mb, int oc) {
        PRAGMA_OMP_SIMD()
        for (int sp = 0; sp < SP; ++sp) {
            auto offset = (size_t)(mb * OC + oc) * SP + sp;
            dst[offset] += bias[oc];
        }
    });
}

template <int blksize>
void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc() {
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper dst_d(conf_.dst_pd());

    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int SP = conf_.OW() * conf_.OH() * conf_.OD();

    const ptrdiff_t stride_mb = dst_d.blocking_desc().strides[0][0];

    parallel_nd(MB, utils::div_up(OC, blksize), SP,
        [&](int mb, int oc_blk, int sp) {
        int oc = oc_blk * blksize;
        auto offset = mb * stride_mb + oc * SP + sp * blksize;
        const int blk = nstl::min(blksize, OC - oc);

        PRAGMA_OMP_SIMD()
        for (int i = 0; i < blk; ++i)
            dst[offset + i] += bias[oc + i];
    });
}

void ref_deconvolution_bwd_weights_t::compute_bwd_bias() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());

    const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OC = conf_.OC() / G;
    const int OD = conf_.OD();
    const int ndims = conf_.desc()->src_desc.ndims;

    parallel_nd(G, OC, [&](int g, int oc) {
        data_t db = 0;
        for (int mb = 0; mb < MB; ++mb) {
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        if (ndims == 5)
                            db += diff_dst[
                                diff_dst_d.off(mb, g*OC + oc, od, oh, ow)];
                        else
                            db += diff_dst[
                                diff_dst_d.off(mb, g*OC + oc, oh, ow)];
                    }
                }
            }
        }
        diff_bias[g * OC + oc] = db;
    });
}

void ref_deconvolution_bwd_weights_t::compute_bwd_bias_ncdhw() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());

    const int OC = conf_.OC();
    const int MB = conf_.MB();
    const int SP = conf_.OH()*conf_.OW()*conf_.OD();

    parallel_nd(OC, [&](int oc) {
        data_t db = 0;
        for (int mb = 0; mb < MB; ++mb) {
            PRAGMA_OMP_SIMD()
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = (size_t)(mb * OC + oc) * SP + sp;
                db += diff_dst[offset];
            }
        }
        diff_bias[oc] = db;
    });
}

template <int blksize>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());

    const int OC = conf_.OC();
    const int MB = conf_.MB();
    const int SP = conf_.OH() * conf_.OW() * conf_.OD();

    const ptrdiff_t stride_mb = diff_dst_d.blocking_desc().strides[0][0];

    parallel_nd(utils::div_up(OC, blksize), [&](int ocb) {
        data_t db[blksize] = {0};

        for (int mb = 0; mb < MB; ++mb) {
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = mb * stride_mb + (ocb * SP + sp) * blksize;

                PRAGMA_OMP_SIMD()
                for (int i = 0; i < blksize; ++i)
                    db[i] += diff_dst[offset+i];
            }
        }

        const int blk = nstl::min(blksize, OC - ocb * blksize);

        PRAGMA_OMP_SIMD()
        for (int i = 0; i < blk; ++i)
            diff_bias[ocb * blksize + i] = db[i];
    });
}

template void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc<8>();
template void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc<16>();
template void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc<8>();
template void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc<16>();

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
