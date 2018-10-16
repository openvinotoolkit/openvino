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

void ref_deconvolution_fwd_t::compute_fwd_bias() {
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OD = conf_.OD();
    const int OC = conf_.OC() / G;
    const int ndims = conf_.desc()->src_desc.ndims;

#   pragma omp parallel for collapse(5) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            auto b = bias[bias_d.off(g*OC + oc)];
                            if (ndims == 5)
                                dst[dst_d.off(mb, g*OC + oc, od, oh, ow)] =
                                    b + dst[dst_d.off(mb, g*OC + oc, od, oh, ow)];
                            else
                                dst[dst_d.off(mb, g*OC + oc, oh, ow)] =
                                    b + dst[dst_d.off(mb, g*OC + oc, oh, ow)];

                        }
                    }
                }
            }
        }
    }
}

void ref_deconvolution_fwd_t::compute_fwd_bias_ncdhw() {
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));
    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int SP = conf_.OW()*conf_.OH()*conf_.OD();
#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; ++mb) {
        for (int oc = 0; oc < OC; ++oc) {
            PRAGMA_OMP_SIMD()
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = ((mb * OC + oc) * SP + sp );
                dst[offset] += bias[oc];
            }
        }
    }
}

template <int blksize>
void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc() {
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));
    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int SP = conf_.OW()*conf_.OH()*conf_.OD();
#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; ++mb) {
        for (int oc = 0; oc < OC/blksize; ++oc) {
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = ((mb * OC + oc*blksize)
                    * SP + sp * blksize) ;

                PRAGMA_OMP_SIMD()
                for (int i=0; i<blksize; i++)
                    dst[offset + i] += bias[oc*blksize + i];
            }
        }
    }
}

void ref_deconvolution_bwd_weights_t::compute_bwd_bias() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OC = conf_.OC() / G;
    const int OD = conf_.OD();
    const int ndims = conf_.desc()->src_desc.ndims;

#   pragma omp parallel for collapse(2) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int oc = 0; oc < OC; ++oc) {
            data_t db = 0;
            for (int mb = 0; mb < MB; ++mb) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            if (ndims == 5)
                                db += diff_dst[diff_dst_d.off(mb, g*OC + oc, od, oh,
                                    ow)];
                            else
                                db += diff_dst[diff_dst_d.off(mb, g*OC + oc, oh,
                                    ow)];

                        }
                    }
                }
            }
            diff_bias[diff_bias_d.off(g*OC+oc)] = db;
       }
    }
}

void ref_deconvolution_bwd_weights_t::compute_bwd_bias_ncdhw() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    const int OC = conf_.OC();
    const int MB = conf_.MB();
    const int SP = conf_.OH()*conf_.OW()*conf_.OD();

#   pragma omp parallel for schedule(static)
    for (int oc = 0; oc < OC; ++oc) {
        data_t db = 0;
        for (int mb = 0; mb < MB; ++mb) {
            PRAGMA_OMP_SIMD()
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = (mb * OC + oc) * SP + sp;
                db += diff_dst[offset];
            }
        }
        diff_bias[oc] = db;
    }
}

template <int blksize>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    const int OC = conf_.OC();
    const int MB = conf_.MB();
    const int SP = conf_.OH()*conf_.OW()*conf_.OD();

#   pragma omp parallel for schedule(static)
    for (int oc = 0; oc < OC/blksize; ++oc) {
        data_t db[blksize] = {0};
        for (int mb = 0; mb < MB; ++mb) {
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = (mb * OC + oc*blksize) * SP + sp * blksize;
                PRAGMA_OMP_SIMD()
                for (int i = 0; i<blksize; i++)
                    db[i] += diff_dst[offset+i];
            }
        }

        PRAGMA_OMP_SIMD()
        for (int i = 0; i<blksize; i++)
            diff_bias[oc*blksize+i] = db[i];
    }
}

template void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc<8>();
template void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc<16>();
template void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc<8>();
template void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc<16>();
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
