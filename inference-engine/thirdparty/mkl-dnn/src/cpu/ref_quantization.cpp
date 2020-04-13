/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <assert.h>
#include <math.h>
#include <common/utils.hpp>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"

#include "ref_quantization.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace alg_kind;
using math::saturate;

template <impl::data_type_t src_type, impl::data_type_t dst_type>
void ref_quantization_fwd_t<src_type, dst_type>::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t*>(this->input_memory(0));

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();

    if (pd()->is_binarization()) {
        auto dst = reinterpret_cast<uint8_t *>(this->memory());

        const int nbits = 8;
        const int CB = utils::div_up(C, nbits);

        auto thresholds = reinterpret_cast<const src_data_t*>(this->input_memory(1));
        auto output_mask = reinterpret_cast<const uint32_t *>(this->input_memory(2));

        const memory_desc_wrapper thresholds_d(pd()->weights_pd(0));
        const memory_desc_wrapper output_mask_d(pd()->weights_pd(1));

        parallel_nd(MB, CB, D, H, W,
            [&](int n, int cb, int d, int h, int w) {
                uint8_t bin_val = 0x00;
                for (int c = cb * nbits, shift = 0; c < std::min(C, (cb + 1) * nbits); c++, shift++) {
                    size_t src_off = src_d.ndims() == 4
                                     ? src_d.off(n, c, h, w)
                                     : src_d.ndims() == 5
                                       ? src_d.off(n, c, d, h, w)
                                       : src_d.off(n, c);

                    size_t thr_off = thresholds_d.off(c);
                    size_t out_mask_off = output_mask_d.off(c);

                    float val = src[src_off];
                    float thr = thresholds[thr_off];
                    uint32_t out_mask = output_mask[out_mask_off];

                    uint32_t res = (val > thr) ? 0xffffffff : 0x00000000;

                    auto bit = uint8_t(res == out_mask);
                    bin_val |= (bit << shift);
                }

                size_t dst_off = dst_d.ndims() == 4
                                 ? dst_d.off(n, cb * nbits, h, w)
                                 : dst_d.ndims() == 5
                                   ? dst_d.off(n, cb, d, h, w)
                                   : dst_d.off(n, cb);

                dst[dst_off / nbits] = bin_val;
        });
    } else {
        auto dst = reinterpret_cast<dst_data_t *>(this->memory());

        auto crop_low = reinterpret_cast<const float *>(this->input_memory(1));
        auto crop_high = reinterpret_cast<const float *>(this->input_memory(2));
        auto input_scale = reinterpret_cast<const float *>(this->input_memory(3));
        auto input_shift = reinterpret_cast<const float *>(this->input_memory(4));
        auto output_scale = reinterpret_cast<const float *>(this->input_memory(5));
        auto output_shift = reinterpret_cast<const float *>(this->input_memory(6));

        const memory_desc_wrapper crop_low_d(pd()->weights_pd(0));
        const memory_desc_wrapper crop_high_d(pd()->weights_pd(1));
        const memory_desc_wrapper input_scale_d(pd()->weights_pd(2));
        const memory_desc_wrapper input_shift_d(pd()->weights_pd(3));
        const memory_desc_wrapper output_scale_d(pd()->weights_pd(4));
        const memory_desc_wrapper output_shift_d(pd()->weights_pd(5));

        parallel_nd(MB, C, D, H, W,
            [&](int n, int c, int d, int h, int w) {
                size_t src_off = src_d.ndims() == 5 ? src_d.off(n, c, d, h, w) :
                                 src_d.ndims() == 4 ? src_d.off(n, c, h, w) :
                                 src_d.ndims() == 3 ? src_d.off(n, c, h) :
                                 src_d.ndims() == 2 ? src_d.off(n, c) :
                                                      src_d.off(n);

                int wei_idx = pd()->axis() == 0 ? n : c;

                size_t crop_low_off = crop_low_d.off(wei_idx);
                size_t crop_high_off = crop_high_d.off(wei_idx);
                size_t input_scale_off = input_scale_d.off(wei_idx);
                size_t input_shift_off = input_shift_d.off(wei_idx);
                size_t output_scale_off = output_scale_d.off(wei_idx);
                size_t output_shift_off = output_shift_d.off(wei_idx);

                float src_val = src[src_off];

                float cl = crop_low[crop_low_off];
                float ch = crop_high[crop_high_off];
                float isc = input_scale[input_scale_off];
                float ish = input_shift[input_shift_off];
                float osc = output_scale[output_scale_off];
                float osh = output_shift[output_shift_off];

                float dst_val = nstl::min(ch, nstl::max(cl, src_val));
                dst_val = dst_val * isc + ish;
                dst_val = roundf(dst_val);
                dst_val = dst_val * osc + osh;

                size_t dst_off = dst_d.ndims() == 5 ? dst_d.off(n, c, d, h, w) :
                                 dst_d.ndims() == 4 ? dst_d.off(n, c, h, w) :
                                 dst_d.ndims() == 3 ? dst_d.off(n, c, h) :
                                 dst_d.ndims() == 2 ? dst_d.off(n, c) :
                                                      dst_d.off(n);

                dst[dst_off] = saturate<dst_data_t>(dst_val);
        });
    }
}

template struct ref_quantization_fwd_t<data_type::f32, data_type::bin>;
template struct ref_quantization_fwd_t<data_type::f32, data_type::u8>;
template struct ref_quantization_fwd_t<data_type::f32, data_type::s8>;
template struct ref_quantization_fwd_t<data_type::f32, data_type::f32>;
template struct ref_quantization_fwd_t<data_type::u8, data_type::u8>;
template struct ref_quantization_fwd_t<data_type::u8, data_type::f32>;

}
}
}
