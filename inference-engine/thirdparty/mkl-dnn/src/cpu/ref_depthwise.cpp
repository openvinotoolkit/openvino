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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "ref_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace alg_kind;

template <typename T> inline T scale_shift_fwd(T s_val, T w_val, T b_val) {
    return s_val*w_val + b_val;
}

template <typename T> inline T prelu_fwd(T s_val, T w_val) {
    return s_val >= 0 ? s_val : s_val*w_val;
}

template <impl::data_type_t data_type>
void ref_depthwise_fwd_t<data_type>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const int MB = conf_.MB();
    const int C = conf_.C();
    const int H = conf_.H();
    const int W = conf_.W();
    const auto alg_kind = conf_.desc()->alg_kind;

    #pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < MB; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    size_t data_off = data_d.ndims() == 4
                                    ? data_d.off(n, c, h, w)
                                    : data_d.off(n, c);

                    data_t s_val = src[data_off];
                    data_t w_val = weights[weights_d.off(c)];
                    data_t b_val = bias ? bias[bias_d.off(c)] : (data_t)0;
                    data_t &d_val = dst[data_off];

                    switch (alg_kind) {
                        case depthwise_scale_shift: d_val = scale_shift_fwd(s_val, w_val, b_val); break;
                        case depthwise_prelu: d_val = prelu_fwd(s_val, w_val); break;
                        default: assert(!"unknown depthwise alg_kind");
                    }
                }
            }
        }
    }
}

template struct ref_depthwise_fwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
