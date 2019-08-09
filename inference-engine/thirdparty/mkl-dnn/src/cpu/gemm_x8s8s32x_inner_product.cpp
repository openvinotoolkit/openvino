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

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "simple_q10n.hpp"
#include "gemm_x8s8s32x_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace math;
using namespace memory_format;
using namespace memory_tracking::names;

template <data_type_t src_type, data_type_t dst_type>
void gemm_x8s8s32x_inner_product_fwd_t<src_type, dst_type
        >::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const int MB = pd()->MB();
    const int OC = pd()->OC();

    bool wei_tr = utils::one_of(pd()->weights_pd()->desc()->format,
            oi, oiw, owi, oihw, ohwi, oidhw, odhwi);

    const int M = OC;
    const int N = MB;
    const int K = pd()->IC_total_padded();
    const int8_t off_a = 0, off_b = 0;
    const int32_t off_c = 0;

    const float *scales = pd()->attr()->output_scales_.scales_;

    acc_data_t *acc = pd()->dst_is_acc_
        ? (acc_data_t *)dst
        : scratchpad().template get<acc_data_t>(key_iprod_int_dat_in_acc_dt);

    const float onef = 1.0, zerof = 0.0;

    if (src_type == data_type::u8) {
        mkldnn_gemm_s8u8s32(wei_tr ? "T" : "N", "N", "F", &M, &N, &K, &onef,
                weights, wei_tr ? &K : &M, &off_a, (uint8_t *)src, &K, &off_b, &zerof,
                acc, &M, &off_c);
    } else if (src_type == data_type::s8) {
        mkldnn_gemm_s8s8s32(wei_tr ? "T" : "N", "N", "F", &M, &N, &K, &onef,
                weights, wei_tr ? &K : &M, &off_a, (int8_t *)src, &K, &off_b, &zerof,
                acc, &M, &off_c);
    } else {
        assert(!"incorrect src type");
    }

    if (!pd()->attr()->has_default_values() || !pd()->dst_is_acc_
            || pd()->with_bias()) {
        const bool force_sequential = MB * OC < 2000;
        parallel(force_sequential ? 1 : 0, (size_t)OC * MB, [&](int ithr, int nthr) {
            size_t start = 0, end = 0;
            balance211((size_t)OC * MB, nthr, ithr, start, end);
            (*pp_kernel_)(dst, acc, bias, scales, start, end);
        });
    }
}

using namespace data_type;

template struct gemm_x8s8s32x_inner_product_fwd_t<u8, f32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<u8, s32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<u8, s8>;
template struct gemm_x8s8s32x_inner_product_fwd_t<u8, u8>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, f32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, s32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, s8>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, u8>;
}
}
}
