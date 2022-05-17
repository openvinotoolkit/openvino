/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "cpu/cpu_engine.hpp"

#include "cpu/ref_lrn.hpp"

#if DNNL_X64
#include "cpu/x64/lrn/jit_avx512_common_lrn.hpp"
#include "cpu/x64/lrn/jit_uni_lrn.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const impl_list_item_t impl_list[] = {
        REG_LRN_P_FWD(CPU_INSTANCE_X64(jit_avx512_common_lrn_fwd_t, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_LRN_P_BWD(CPU_INSTANCE_X64(jit_avx512_common_lrn_bwd_t, f32))
#endif
        REG_LRN_P_FWD(CPU_INSTANCE_X64(jit_avx512_common_lrn_fwd_t, bf16))
#ifdef ENABLE_UNUSED_PRIM
        REG_LRN_P_BWD(CPU_INSTANCE_X64(jit_avx512_common_lrn_bwd_t, bf16))
#endif
        REG_LRN_P_FWD(CPU_INSTANCE_X64(jit_uni_lrn_fwd_t, avx512_common, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_LRN_P_BWD(CPU_INSTANCE_X64(jit_uni_lrn_bwd_t, avx512_common, f32))
#endif
        REG_LRN_P_FWD(CPU_INSTANCE_X64(jit_uni_lrn_fwd_t, avx512_common, bf16))
#ifdef ENABLE_UNUSED_PRIM
        REG_LRN_P_BWD(CPU_INSTANCE_X64(jit_uni_lrn_bwd_t, avx512_common, bf16))
#endif
        REG_LRN_P_FWD(CPU_INSTANCE_X64(jit_uni_lrn_fwd_t, avx2, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_LRN_P_BWD(CPU_INSTANCE_X64(jit_uni_lrn_bwd_t, avx2, f32))
#endif
        REG_LRN_P_FWD(CPU_INSTANCE_X64(jit_uni_lrn_fwd_t, sse41, f32))
        REG_LRN_P_FWD(CPU_INSTANCE(ref_lrn_fwd_t, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_LRN_P_BWD(CPU_INSTANCE(ref_lrn_bwd_t, f32))
        REG_LRN_P_FWD(CPU_INSTANCE(ref_lrn_fwd_t, bf16))
        REG_LRN_P_BWD(CPU_INSTANCE(ref_lrn_bwd_t, bf16))
#endif
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const impl_list_item_t *get_lrn_impl_list(const lrn_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
