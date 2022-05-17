/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
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

#include "cpu/ref_softmax.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_softmax.hpp"
#include "cpu/x64/jit_uni_fork_softmax.hpp"
using namespace dnnl::impl::cpu::x64;
#elif DNNL_AARCH64
#include "cpu/aarch64/jit_uni_softmax.hpp"
using namespace dnnl::impl::cpu::aarch64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const impl_list_item_t impl_list[] = {
        REG_SOFTMAX_P_FWD(CPU_INSTANCE_X64(jit_uni_softmax_fwd_t, avx512_common))
#ifdef ENABLE_UNUSED_PRIM
        REG_SOFTMAX_P_BWD(CPU_INSTANCE_X64(jit_uni_softmax_bwd_t, avx512_common))
#endif
        REG_SOFTMAX_P_FWD(CPU_INSTANCE_X64(jit_uni_softmax_fwd_t, avx2))
        REG_SOFTMAX_P_FWD(CPU_INSTANCE_X64(jit_uni_softmax_fwd_t, sse41))
        REG_SOFTMAX_P_FWD(CPU_INSTANCE_X64(jit_uni_fork_softmax_fwd_t, avx512_common))
        REG_SOFTMAX_P_FWD(CPU_INSTANCE_X64(jit_uni_fork_softmax_fwd_t, avx2))
        REG_SOFTMAX_P_FWD(CPU_INSTANCE_X64(jit_uni_fork_softmax_fwd_t, sse41))
#ifdef ENABLE_UNUSED_PRIM
        CPU_INSTANCE_AARCH64(jit_uni_softmax_fwd_t, sve_512)
        CPU_INSTANCE_AARCH64(jit_uni_softmax_bwd_t, sve_512)
#endif
        REG_SOFTMAX_P_FWD(CPU_INSTANCE(ref_softmax_fwd_t, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_SOFTMAX_P_BWD(CPU_INSTANCE(ref_softmax_bwd_t, f32))
        REG_SOFTMAX_P_FWD(CPU_INSTANCE(ref_softmax_fwd_t, bf16))
        REG_SOFTMAX_P_BWD(CPU_INSTANCE(ref_softmax_bwd_t, bf16))
#endif
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const impl_list_item_t *get_softmax_impl_list(const softmax_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

const impl_list_item_t *get_logsoftmax_impl_list(
        const logsoftmax_desc_t *desc) {
    return get_softmax_impl_list(desc);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
