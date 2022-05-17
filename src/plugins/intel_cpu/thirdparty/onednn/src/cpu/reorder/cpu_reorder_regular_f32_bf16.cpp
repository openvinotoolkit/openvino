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

const impl_list_map_t regular_f32_bf16_impl_list_map {
    // f32 -> bf16
    {{f32, bf16, 0}, {
        REG_RNN_P_FWD(CPU_REORDER_INSTANCE(rnn_weights_reorder_t, f32, bf16))

        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_blk_reorder_t)))
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t)))

        REG_REORDER_P(REG_SR_BIDIR(f32, ncw, bf16, nCw16c))
        REG_REORDER_P(REG_SR_BIDIR(f32, nchw, bf16, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(f32, any, bf16, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(f32, any, bf16, nCdhw16c))

        REG_REORDER_P(REG_SR(f32, oihw, bf16, OIhw8i16o2i, fmt_order_keep))
        REG_REORDER_P(REG_SR(f32, goihw, bf16, gOIhw8i16o2i, fmt_order_keep))
        REG_REORDER_P(REG_SR(f32, oihw, bf16, OIhw8o16i2o, fmt_order_keep))
        REG_REORDER_P(REG_SR(f32, goihw, bf16, gOIhw8o16i2o, fmt_order_keep))
        REG_REORDER_P(REG_SR(f32, oihw, bf16, IOhw8o16i2o, fmt_order_keep))
        REG_REORDER_P(REG_SR(f32, goihw, bf16, gIOhw8o16i2o, fmt_order_keep))
        REG_REORDER_P(REG_SR(f32, oihw, bf16, OIhw16i16o, fmt_order_keep))
        REG_REORDER_P(REG_SR(f32, goihw, bf16, gOIhw16i16o, fmt_order_keep))

        REG_REORDER_P(REG_SR(f32, any, bf16, any, fmt_order_any, spec_reference))

        nullptr,
    }},
};

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
