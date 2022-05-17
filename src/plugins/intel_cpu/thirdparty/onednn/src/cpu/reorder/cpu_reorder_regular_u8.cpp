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

#ifdef __INTEL_COMPILER
        /* Enable direct copy primitives for non-icc compilers, but place it after the jitted ones */
#define REG_FAST_DIRECT_COPY_AFTER_JIT(sdt, ddt)
#else
#define REG_FAST_DIRECT_COPY_AFTER_JIT(sdt, ddt) REG_SR_DIRECT_COPY(sdt, ddt)
#endif

// clang-format off

const impl_list_map_t regular_u8_impl_list_map {
    // u8 ->
    {{u8, data_type::undef, 0}, {
        REG_REORDER_P(REG_FAST_DIRECT_COPY(u8, f32))
        REG_REORDER_P(REG_FAST_DIRECT_COPY(u8, s32))
        REG_REORDER_P(REG_FAST_DIRECT_COPY(u8, bf16))
        REG_REORDER_P(REG_FAST_DIRECT_COPY(u8, s8))
        REG_REORDER_P(REG_FAST_DIRECT_COPY(u8, u8))

        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_blk_reorder_t)))
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t)))

        DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64_jit_uni_reorder_t))

        // Allow direct-copy primitives for non-intel compilers, but with a lower priority than the jitted impl
        REG_REORDER_P(REG_FAST_DIRECT_COPY_AFTER_JIT(u8, f32))
        REG_REORDER_P(REG_FAST_DIRECT_COPY_AFTER_JIT(u8, s32))
        REG_REORDER_P(REG_FAST_DIRECT_COPY_AFTER_JIT(u8, bf16))
        REG_REORDER_P(REG_FAST_DIRECT_COPY_AFTER_JIT(u8, s8))
        REG_REORDER_P(REG_FAST_DIRECT_COPY_AFTER_JIT(u8, u8))

        REG_REORDER_P(REG_SR_BIDIR(u8, any, f32, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(u8, any, s32, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(u8, any, bf16, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(u8, any, s8, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(u8, any, u8, nChw16c))
        REG_REORDER_P(REG_SR_BIDIR(u8, any, f32, nChw8c))
        REG_REORDER_P(REG_SR_BIDIR(u8, any, s32, nChw8c))
        REG_REORDER_P(REG_SR_BIDIR(u8, any, s8, nChw8c))
        REG_REORDER_P(REG_SR_BIDIR(u8, any, u8, nChw8c))

        REG_REORDER_P(REG_SR(u8, any, f32, any, fmt_order_any, spec_reference))
        REG_REORDER_P(REG_SR(u8, any, s32, any, fmt_order_any, spec_reference))
        REG_REORDER_P(REG_SR(u8, any, bf16, any, fmt_order_any, spec_reference))
        REG_REORDER_P(REG_SR(u8, any, u8, any, fmt_order_any, spec_reference))
        REG_REORDER_P(REG_SR(u8, any, s8, any, fmt_order_any, spec_reference))

        nullptr,
    }},
};

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
