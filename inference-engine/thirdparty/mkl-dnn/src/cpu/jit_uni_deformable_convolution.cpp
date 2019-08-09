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

#include "mkldnn_types.h"
#include "c_types_map.hpp"
#include "jit_uni_deformable_convolution.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include <cstring>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa>
void jit_uni_deformable_convolution_fwd_t<isa>::execute_forward() const {
    auto src = reinterpret_cast<const float *>(this->input_memory(0));
    auto offsets = reinterpret_cast<const float *>(this->input_memory(1));
    auto weights = reinterpret_cast<const float *>(this->input_memory(2));
    auto bias = reinterpret_cast<const float *>(this->input_memory(3));
    auto dst = reinterpret_cast<float *>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd(0));
    const memory_desc_wrapper offsets_d(pd()->src_pd(1));
    const memory_desc_wrapper dst_d(pd()->dst_pd());
//    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const auto &jcp = kernel_->jcp;

    if (bias && jcp.oc != jcp.oc_padded) {
        auto padded_bias = this->scratchpad().template get<float>(key_conv_padded_bias);
        utils::array_copy(padded_bias, (float*)bias, jcp.oc);
        utils::array_set(padded_bias + jcp.oc, 0, jcp.oc_padded - jcp.oc);
        bias = (float *)padded_bias;
    }

    auto input_buffer = this->scratchpad().template get<float>(key_def_conv_buffer);

    const size_t work_amount = jcp.mb * jcp.ngroups * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, g{0}, oh{0};
        nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
            auto par_conv = jit_def_conv_call_s();

            const size_t _oc = g * jcp.nb_oc;
            const size_t _ic = g * jcp.nb_ic;

            par_conv.src = &src[src_d.blk_off(n, _ic*jcp.ic_block, oh * jcp.stride_h - jcp.t_pad, 0 - jcp.l_pad)];
            par_conv.off = &offsets[offsets_d.blk_off(n, 0, oh, 0)];
            par_conv.filt = weights;//weights_d(0, 0, 0, 0);
            if (bias)
                par_conv.bias = &bias[bias_d.blk_off(_oc * jcp.oc_block*jcp.typesize_bia)];
            par_conv.dst = &dst[dst_d.blk_off(n, _oc*jcp.oc_block, oh, 0)];

            par_conv.buf = input_buffer + ithr * jcp.ur_w * jcp.kh * jcp.kw * jcp.ic;

            par_conv.oh_pos = oh;

            kernel_->jit_ker(&par_conv);
            nd_iterator_step(n, jcp.mb, g, jcp.ngroups, oh, jcp.oh);
        }
    };

    parallel(0, ker);
}

template struct jit_uni_deformable_convolution_fwd_t<avx512_common>;
template struct jit_uni_deformable_convolution_fwd_t<avx2>;
template struct jit_uni_deformable_convolution_fwd_t<sse42>;

}
}
}
