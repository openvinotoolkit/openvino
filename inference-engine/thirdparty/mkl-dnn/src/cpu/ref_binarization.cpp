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

#include "ref_binarization.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace alg_kind;

template <impl::data_type_t src_type>
void ref_binarization_fwd_t<src_type>::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t*>(this->input_memory(0));
    auto weights = reinterpret_cast<const src_data_t*>(this->input_memory(1));
    auto output_mask = reinterpret_cast<const uint32_t*>(this->input_memory(2));
    auto dst = reinterpret_cast<uint8_t*>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper output_mask_d(pd()->weights_pd(1));

    int nbits = 8;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int CB = utils::div_up(C, nbits);
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();

    parallel_nd(MB, CB, D, H, W,
        [&](int n, int cb, int d, int h, int w) {

        uint8_t bin_val = 0x00;
        for (int c = cb * nbits, shift = 0; c < std::min(C, (cb + 1) * nbits); c++, shift++) {
            size_t src_off = src_d.ndims() == 4
                              ? src_d.off(n, c, h, w)
                              : src_d.ndims() == 5
                                ? src_d.off(n, c, d, h, w)
                                : src_d.off(n, c);

            size_t wei_off = weights_d.off(c);
            size_t out_mask_off = output_mask_d.off(c);

            float val = src[src_off];
            float thr = weights[wei_off];
            uint32_t out_mask = output_mask[out_mask_off];

            uint32_t res = (val > thr) ? 0xffffffff : 0x00000000;

            auto bit = uint8_t(res == out_mask);
            bin_val |= (bit << shift);
        }

        size_t dst_off = dst_d.ndims() == 4
                           ? dst_d.off(n, cb*nbits, h, w)
                           : dst_d.ndims() == 5
                             ? dst_d.off(n, cb, d, h, w)
                             : dst_d.off(n, cb);

        dst[dst_off / nbits] = bin_val;
    });
}

template struct ref_binarization_fwd_t<data_type::f32>;

}
}
}
