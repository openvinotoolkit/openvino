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

#include "cpu/ref_deconvolution.hpp"

#if DNNL_X64
#include "cpu/x64/jit_avx512_core_amx_deconvolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_1x1_deconvolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_deconvolution.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_1x1_deconvolution.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_deconvolution.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const impl_list_item_t impl_list[] = {
#ifdef ENABLE_UNUSED_PRIM
        REG_DECONV_P_FWD(CPU_INSTANCE_X64(jit_avx512_core_amx_deconvolution_fwd_t))
#endif
        REG_DECONV_P_FWD(CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t))
        REG_DECONV_P_FWD(CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_deconvolution_fwd_t))
        REG_DECONV_P_FWD(CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t, avx2))
        REG_DECONV_P_FWD(CPU_INSTANCE_X64(jit_uni_x8s8s32x_deconvolution_fwd_t, avx2))
        REG_DECONV_P_FWD(CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t, sse41))
        REG_DECONV_P_FWD(CPU_INSTANCE_X64(jit_uni_x8s8s32x_deconvolution_fwd_t, sse41))
#ifdef ENABLE_UNUSED_PRIM
        REG_DECONV_P_BWD(CPU_INSTANCE(ref_deconvolution_bwd_weights_t))
        REG_DECONV_P_BWD(CPU_INSTANCE(ref_deconvolution_bwd_data_t))
        REG_DECONV_P_FWD(CPU_INSTANCE(ref_deconvolution_fwd_t))
#endif
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const impl_list_item_t *get_deconvolution_impl_list(
        const deconvolution_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
