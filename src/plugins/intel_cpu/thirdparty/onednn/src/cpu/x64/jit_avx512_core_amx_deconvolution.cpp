/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/jit_avx512_core_amx_conv_utils.hpp"
#include "cpu/x64/jit_avx512_core_amx_deconvolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::memory_tracking::names;

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

// NOTE: This primitive shares a kernel with bwd/d convolution. Hence, all
//       parameters stored in `pd()->jcp_` are in terms of bwd/d convolution.
//       This means that the following parameters have been exchanged:
//         1. ic <-> oc
//         2. ih <-> oh
//         3. iw <-> ow
//       The same exchange applies to all derivative values in `pd()->jcp_`
//       (eg, ic_block <-> oc_block, etc).

void jit_avx512_core_amx_deconvolution_fwd_t::prepare_padded_bias(
        const char *&bias, const memory_tracking::grantor_t &scratchpad) const {
    auto &jcp = pd()->jcp_;
    if (jcp.with_bias && jcp.ic != jcp.ic_without_padding) {
        const size_t bia_dt_size = jcp.typesize_bia;
        auto padded_bias = scratchpad.template get<char>(
                memory_tracking::names::key_conv_padded_bias);
        utils::array_copy(
                padded_bias, bias, bia_dt_size * jcp.ic_without_padding);
        utils::array_set(padded_bias + bia_dt_size * jcp.ic_without_padding,
                0.f, bia_dt_size * (jcp.ic - jcp.ic_without_padding));
        bias = padded_bias;
    }
}

void jit_avx512_core_amx_deconvolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const float *oscales = pd()->attr()->output_scales_.scales_;

    // The body of bwd/d convolution harness is called with:
    //   1. src as input instead of diff_dst
    //   2. dst as output instead of diff_src
    amx_utils::execute_backward_convolution_body(ctx, pd()->jcp_, kernel_, src,
            weights, bias, oscales, dst, src_d, weights_d, bias_d, dst_d);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
