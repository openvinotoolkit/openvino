/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "cpu/reorder/cpu_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// clang-format off
const impl_list_map_t regular_bf16_impl_list_map {
    // bf16 ->
    {{bf16, data_type::undef, 0}, {
        REG_RNN_P_FWD(CPU_REORDER_INSTANCE(rnn_weights_reorder_t, bf16, bf16))

        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_blk_reorder_t)))
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t)))

        REG_REORDER_P(REG_SR_BIDIR(bf16, any, f32, nCw16c))
        REG_REORDER_P(REG_SR_BIDIR(bf16, any, f32, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(bf16, any, f32, nCdhw16c))

        REG_REORDER_P(REG_SR_BIDIR(bf16, any, s8, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(bf16, any, s8, nCdhw16c))

        REG_REORDER_P(REG_SR_BIDIR(bf16, any, u8, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(bf16, any, u8, nCdhw16c))

        REG_REORDER_P(REG_SR_BIDIR(bf16, any, bf16, nCw16c))
        REG_REORDER_P(REG_SR_BIDIR(bf16, any, bf16, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(bf16, any, bf16, nCdhw16c))

        REG_REORDER_P(REG_SR_BIDIR(bf16, any, f32, OIdhw16o16i))
        REG_REORDER_P(REG_SR_BIDIR(bf16, any, f32, OIdhw16i16o))

        REG_REORDER_P(REG_SR_BIDIR(bf16, any, s8, OIdhw16o16i))
        REG_REORDER_P(REG_SR_BIDIR(bf16, any, s8, OIdhw16i16o))

        REG_REORDER_P(REG_SR_BIDIR(bf16, any, u8, OIdhw16o16i))
        REG_REORDER_P(REG_SR_BIDIR(bf16, any, u8, OIdhw16i16o))

        REG_REORDER_P(REG_SR(bf16, any, bf16, any, fmt_order_any, spec_reference))
        REG_REORDER_P(REG_SR(bf16, any, f32, any, fmt_order_any, spec_reference))
        REG_REORDER_P(REG_SR(bf16, any, s8, any, fmt_order_any, spec_reference))
        REG_REORDER_P(REG_SR(bf16, any, u8, any, fmt_order_any, spec_reference))

        nullptr,
    }},
};

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
