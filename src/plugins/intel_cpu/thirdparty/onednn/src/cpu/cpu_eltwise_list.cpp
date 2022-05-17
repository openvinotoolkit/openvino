/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
* Copyright 2021 Arm Ltd. and affiliates
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

#include "cpu/ref_eltwise.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "cpu/x64/jit_uni_eltwise_int.hpp"
using namespace dnnl::impl::cpu::x64;
#elif DNNL_AARCH64
#include "cpu/aarch64/jit_uni_eltwise.hpp"
#include "cpu/aarch64/jit_uni_eltwise_int.hpp"
#if DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/acl_eltwise.hpp"
#endif // DNNL_AARCH64_USE_ACL
using namespace dnnl::impl::cpu::aarch64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const impl_list_item_t impl_list[] = {
#ifdef ENABLE_UNUSED_PRIM
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, avx512_common, f32))
        REG_ELTWISE_P_BWD(CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, avx512_common, f32))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, avx512_core, bf16))
        REG_ELTWISE_P_BWD(CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, avx512_core, bf16))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, avx2, f32))
        REG_ELTWISE_P_BWD(CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, avx2, f32))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, avx, f32))
        REG_ELTWISE_P_BWD(CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, avx, f32))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, sse41, f32))
        REG_ELTWISE_P_BWD(CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, sse41, f32))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx512_common, s32))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx512_common, s8))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx512_common, u8))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx2, s32))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx2, s8))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx2, u8))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, sse41, s32))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, sse41, s8))
        REG_ELTWISE_P_FWD(CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, sse41, u8))
        CPU_INSTANCE_AARCH64(jit_uni_eltwise_fwd_t, sve_512, f32)
        CPU_INSTANCE_AARCH64(jit_uni_eltwise_bwd_t, sve_512, f32)
        CPU_INSTANCE_AARCH64(jit_uni_eltwise_int_fwd_t, sve_512, s32)
        CPU_INSTANCE_AARCH64(jit_uni_eltwise_int_fwd_t, sve_512, s8)
        CPU_INSTANCE_AARCH64(jit_uni_eltwise_int_fwd_t, sve_512, u8)
        CPU_INSTANCE_AARCH64_ACL(acl_eltwise_fwd_t, f32)
        CPU_INSTANCE_AARCH64_ACL(acl_eltwise_fwd_t, s8)
        REG_ELTWISE_P_FWD(CPU_INSTANCE(ref_eltwise_fwd_t, f32))
        REG_ELTWISE_P_BWD(CPU_INSTANCE(ref_eltwise_bwd_t, f32))
        REG_ELTWISE_P_FWD(CPU_INSTANCE(ref_eltwise_fwd_t, bf16))
        REG_ELTWISE_P_BWD(CPU_INSTANCE(ref_eltwise_bwd_t, bf16))
        REG_ELTWISE_P_FWD(CPU_INSTANCE(ref_eltwise_fwd_t, s32))
        REG_ELTWISE_P_FWD(CPU_INSTANCE(ref_eltwise_fwd_t, s8))
        REG_ELTWISE_P_FWD(CPU_INSTANCE(ref_eltwise_fwd_t, u8))
#endif
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const impl_list_item_t *get_eltwise_impl_list(const eltwise_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
