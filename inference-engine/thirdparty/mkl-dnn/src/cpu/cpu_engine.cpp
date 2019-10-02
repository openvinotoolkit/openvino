/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "type_helpers.hpp"
#include "verbose.hpp"

#include "cpu_concat.hpp"
#include "cpu_sum.hpp"

#include "cpu/rnn/ref_rnn.hpp"

#include "cpu/jit_avx512_core_x8s8s32x_1x1_convolution.hpp"
#include "cpu/jit_avx512_common_1x1_convolution.hpp"
#include "cpu/jit_avx512_core_fp32_wino_conv_4x3.hpp"
#include "cpu/jit_avx512_common_convolution_winograd.hpp"
#include "cpu/jit_avx512_core_x8s8s32x_convolution.hpp"
#include "cpu/jit_avx512_core_bf16_convolution.hpp"
#include "cpu/jit_avx512_core_bf16_1x1_convolution.hpp"
#include "cpu/jit_avx512_common_convolution.hpp"
#include "cpu/jit_avx2_1x1_convolution.hpp"
#include "cpu/jit_sse42_1x1_convolution.hpp"
#include "cpu/jit_avx2_convolution.hpp"
#include "cpu/jit_sse42_convolution.hpp"
#include "cpu/gemm_convolution.hpp"
#include "cpu/gemm_bf16_convolution.hpp"
#include "cpu/gemm_x8s8s32x_convolution.hpp"
#include "cpu/ref_convolution.hpp"
#include "cpu/jit_avx512_core_x8s8s32x_deconvolution.hpp"
#include "cpu/jit_avx512_core_x8s8s32x_1x1_deconvolution.hpp"
#include "cpu/ref_deconvolution.hpp"
#include "cpu/ref_shuffle.hpp"
#include "cpu/jit_uni_eltwise.hpp"
#include "cpu/ref_eltwise.hpp"
#include "cpu/ref_softmax.hpp"
#include "cpu/jit_uni_pooling.hpp"
#include "cpu/jit_uni_i8i8_pooling.hpp"
#include "cpu/ref_pooling.hpp"
#include "cpu/nchw_pooling.hpp"
#include "cpu/nhwc_pooling.hpp"
#include "cpu/jit_avx512_common_lrn.hpp"
#include "cpu/jit_uni_lrn.hpp"
#include "cpu/ref_lrn.hpp"
#include "cpu/jit_uni_batch_normalization.hpp"
#include "cpu/ref_batch_normalization.hpp"
#include "cpu/ncsp_batch_normalization.hpp"
#include "cpu/nspc_batch_normalization.hpp"
#include "cpu/ref_inner_product.hpp"
#include "cpu/gemm_inner_product.hpp"
#include "cpu/gemm_bf16_inner_product.hpp"
#include "cpu/gemm_x8s8s32x_inner_product.hpp"
#include "cpu/jit_uni_dw_convolution.hpp"
#include "cpu/jit_avx512_core_u8s8s32x_wino_convolution.hpp"
#include "cpu/jit_avx512_core_fp32_wino_conv_2x3.hpp"
#include "cpu/jit_uni_batch_normalization_s8.hpp"
#include "cpu/jit_uni_roi_pooling.hpp"
#include "cpu/jit_uni_softmax.hpp"
#include "cpu/ref_roi_pooling.hpp"
#include "cpu/jit_uni_depthwise.hpp"
#include "cpu/ref_depthwise.hpp"
#include "cpu/jit_uni_x8s8s32x_convolution.hpp"
#include "cpu/jit_uni_x8s8s32x_dw_convolution.hpp"
#include "cpu/jit_sse42_i8i8_pooling.hpp"
#include "cpu/jit_uni_planar_convolution.hpp"
#include "cpu/jit_uni_binary_convolution.hpp"
#include "cpu/ref_binary_convolution.hpp"
#include "cpu/jit_uni_binarization.hpp"
#include "cpu/ref_binarization.hpp"
#include "cpu/jit_uni_deformable_convolution.hpp"
#include "cpu/ref_deformable_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;

status_t cpu_engine_t::memory_primitive_desc_create(memory_pd_t **pd,
        const memory_desc_t *desc) {
    return safe_ptr_assign<memory_pd_t>(*pd,
            new cpu_memory_t::pd_t(this, desc));
}

status_t cpu_engine_t::view_primitive_desc_create(view_pd_t **view_pd,
            const memory_pd_t *memory_pd, const dims_t dims,
            const dims_t offsets) {
    assert(memory_pd->engine() == this);
    cpu_view_t::pd_t *cpu_vpd = nullptr;
    status_t status = cpu_view_t::pd_t::create(&cpu_vpd,
            (const cpu_memory_t::pd_t *)memory_pd, dims, offsets);
    if (status != success) return status;
    *view_pd = cpu_vpd;
    return success;
}

using pd_create_f = mkldnn::impl::engine_t::primitive_desc_create_f;

namespace {
using namespace mkldnn::impl::data_type;

#define INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
static const pd_create_f cpu_impl_list[] = {
    /* RNN */
    INSTANCE(ref_rnn_fwd_f32_t),
    INSTANCE(ref_rnn_fwd_u8s8_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_rnn_bwd_f32_t),
#endif
    /* conv */
    INSTANCE(jit_avx512_common_planar_convolution_fwd_t),
    INSTANCE(jit_avx512_common_dw_convolution_fwd_t),
    INSTANCE(jit_avx512_common_dw_convolution_bwd_data_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx512_common_dw_convolution_bwd_weights_t),
#endif
    INSTANCE(jit_avx512_common_1x1_convolution_fwd_f32_t),
    INSTANCE(jit_avx512_common_1x1_convolution_bwd_data_f32_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx512_common_1x1_convolution_bwd_weights_t),
    INSTANCE(jit_avx512_common_1x1_convolution_fwd_s16s16s32_t),
    INSTANCE(jit_avx512_common_1x1_convolution_bwd_data_s16s16s32_t),
#endif
    INSTANCE(jit_avx512_core_fp32_wino_conv_2x3_fwd_t),
    INSTANCE(jit_avx512_core_fp32_wino_conv_4x3_fwd_t),
    INSTANCE(jit_avx512_core_fp32_wino_conv_4x3_bwd_data_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_t),
#endif
    INSTANCE(jit_avx512_common_convolution_winograd_fwd_t),
    INSTANCE(jit_avx512_common_convolution_winograd_bwd_data_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx512_common_convolution_winograd_bwd_weights_t),
#endif
    INSTANCE(jit_avx512_common_convolution_fwd_t<f32>),
    INSTANCE(jit_avx512_common_convolution_bwd_data_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx512_common_convolution_bwd_weights_t<f32>),
#endif
    INSTANCE(jit_avx2_planar_convolution_fwd_t),
    INSTANCE(jit_avx2_dw_convolution_fwd_t),
    INSTANCE(jit_avx2_dw_convolution_bwd_data_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx2_dw_convolution_bwd_weights_t),
#endif
    INSTANCE(jit_avx2_1x1_convolution_fwd_t),
    INSTANCE(jit_avx2_1x1_convolution_bwd_data_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx2_1x1_convolution_bwd_weights_t),
#endif
    INSTANCE(jit_sse42_dw_convolution_fwd_t),
    INSTANCE(jit_sse42_dw_convolution_bwd_data_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_sse42_dw_convolution_bwd_weights_t),
#endif
    INSTANCE(jit_sse42_1x1_convolution_fwd_t),
    INSTANCE(jit_avx2_convolution_fwd_t),
    INSTANCE(jit_avx2_convolution_bwd_data_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx2_convolution_bwd_weights_t),
#endif
    INSTANCE(jit_sse42_convolution_fwd_t),
    INSTANCE(gemm_convolution_fwd_t),
    INSTANCE(gemm_convolution_bwd_data_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(gemm_convolution_bwd_weights_t),
#endif
    INSTANCE(ref_convolution_fwd_t<f32>),
    INSTANCE(ref_convolution_bwd_data_t<f32, f32, f32, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_convolution_bwd_weights_t<f32, f32, f32, f32>),
    /* conv (bfloat16) */
    INSTANCE(_jit_uni_dw_convolution_fwd_t<avx512_core, bf16, bf16>),
    INSTANCE(_jit_uni_dw_convolution_fwd_t<avx512_core, bf16, f32>),
    INSTANCE(_jit_uni_dw_convolution_bwd_data_t<avx512_core, bf16, bf16>),
    INSTANCE(_jit_uni_dw_convolution_bwd_data_t<avx512_core, bf16, f32>),
    INSTANCE(_jit_uni_dw_convolution_bwd_weights_t<avx512_core, bf16, bf16>),
    INSTANCE(_jit_uni_dw_convolution_bwd_weights_t<avx512_core, bf16, f32>),
    INSTANCE(jit_avx512_core_bf16_1x1_convolution_fwd_t<f32>),
    INSTANCE(jit_avx512_core_bf16_1x1_convolution_fwd_t<bf16>),
    INSTANCE(jit_avx512_core_bf16_1x1_convolution_bwd_data_t<f32>),
    INSTANCE(jit_avx512_core_bf16_1x1_convolution_bwd_data_t<bf16>),
    INSTANCE(jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<f32>),
    INSTANCE(jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<bf16>),
    INSTANCE(jit_avx512_core_bf16_convolution_fwd_t<f32>),
    INSTANCE(jit_avx512_core_bf16_convolution_fwd_t<bf16>),
    INSTANCE(jit_avx512_core_bf16_convolution_bwd_data_t<f32>),
    INSTANCE(jit_avx512_core_bf16_convolution_bwd_data_t<bf16>),
    INSTANCE(jit_avx512_core_bf16_convolution_bwd_weights_t<bf16>),
    INSTANCE(jit_avx512_core_bf16_convolution_bwd_weights_t<f32>),
    INSTANCE(gemm_bf16_convolution_fwd_t<f32>),
    INSTANCE(gemm_bf16_convolution_fwd_t<bf16>),
    INSTANCE(gemm_bf16_convolution_bwd_data_t<f32>),
    INSTANCE(gemm_bf16_convolution_bwd_data_t<bf16>),
    INSTANCE(gemm_bf16_convolution_bwd_weights_t<f32>),
    INSTANCE(gemm_bf16_convolution_bwd_weights_t<bf16>),
#endif
    /* conv (int) */
    INSTANCE(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<f32>),
    INSTANCE(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<s32>),
    INSTANCE(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<s8>),
    INSTANCE(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<u8>),
    INSTANCE(jit_avx512_common_convolution_fwd_t<s16, s16, s32>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8,f32>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8,s32>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8,u8>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8,s8>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8,f32>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8,s32>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8,u8>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8,s8>),
    INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8,f32>),
    INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8,s32>),
    INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8,u8>),
    INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8,s8>),
    INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8,f32>),
    INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8,s32>),
    INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8,u8>),
    INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8,s8>),
    INSTANCE(jit_avx512_common_convolution_bwd_data_t<s16, s16, s32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx512_common_convolution_bwd_weights_t<s16, s16, s32>),
#endif
    INSTANCE(jit_avx2_x8s8s32x_dw_convolution_fwd_t<u8,f32>),
    INSTANCE(jit_avx2_x8s8s32x_dw_convolution_fwd_t<u8,s32>),
    INSTANCE(jit_avx2_x8s8s32x_dw_convolution_fwd_t<u8,u8>),
    INSTANCE(jit_avx2_x8s8s32x_dw_convolution_fwd_t<u8,s8>),
    INSTANCE(jit_sse42_x8s8s32x_dw_convolution_fwd_t<u8,f32>),
    INSTANCE(jit_sse42_x8s8s32x_dw_convolution_fwd_t<u8,s32>),
    INSTANCE(jit_sse42_x8s8s32x_dw_convolution_fwd_t<u8,u8>),
    INSTANCE(jit_sse42_x8s8s32x_dw_convolution_fwd_t<u8,s8>),
    INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<u8,f32>),
    INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<u8,s32>),
    INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<u8,u8>),
    INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<u8,s8>),
    INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<s8,f32>),
    INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<s8,s32>),
    INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<s8,u8>),
    INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<s8,s8>),
    INSTANCE(jit_sse42_x8s8s32x_convolution_fwd_t<u8,f32>),
    INSTANCE(jit_sse42_x8s8s32x_convolution_fwd_t<u8,s32>),
    INSTANCE(jit_sse42_x8s8s32x_convolution_fwd_t<u8,u8>),
    INSTANCE(jit_sse42_x8s8s32x_convolution_fwd_t<u8,s8>),
    INSTANCE(jit_sse42_x8s8s32x_convolution_fwd_t<s8,f32>),
    INSTANCE(jit_sse42_x8s8s32x_convolution_fwd_t<s8,s32>),
    INSTANCE(jit_sse42_x8s8s32x_convolution_fwd_t<s8,u8>),
    INSTANCE(jit_sse42_x8s8s32x_convolution_fwd_t<s8,s8>),
    INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, s32>),
    INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, u8>),
    INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, s8>),
    INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, f32>),
    INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, s32>),
    INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, u8>),
    INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, s8>),
    INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, f32>),
    INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<s32>),
    INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<u8>),
    INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<s8>),
    INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<f32>),
    INSTANCE(ref_convolution_fwd_t<s16, s16, s32, s32>),
    INSTANCE(ref_convolution_fwd_t<u8, s8, f32, s32>),
    INSTANCE(ref_convolution_fwd_t<u8, s8, s32, s32>),
    INSTANCE(ref_convolution_fwd_t<u8, s8, s8, s32>),
    INSTANCE(ref_convolution_fwd_t<u8, s8, u8, s32>),
    INSTANCE(ref_convolution_bwd_data_t<s32, s16, s16, s32>),
    INSTANCE(ref_convolution_bwd_data_t<f32, s8, u8, s32>),
    INSTANCE(ref_convolution_bwd_data_t<s32, s8, u8, s32>),
    INSTANCE(ref_convolution_bwd_data_t<s8, s8, u8, s32>),
    INSTANCE(ref_convolution_bwd_data_t<u8, s8, u8, s32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_convolution_bwd_weights_t<s16, s32, s16, s32>),
#endif
    /* deconv */
#if ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<u8,f32>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<u8,s32>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<u8,u8>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<u8,s8>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<s8,f32>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<s8,s32>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<s8,u8>),
    INSTANCE(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<s8,s8>),
    INSTANCE(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<u8,s32>),
    INSTANCE(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<u8,u8>),
    INSTANCE(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<u8,s8>),
    INSTANCE(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<u8,f32>),
    INSTANCE(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<s8,s32>),
    INSTANCE(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<s8,u8>),
    INSTANCE(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<s8,s8>),
    INSTANCE(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<s8,f32>),
    INSTANCE(ref_deconvolution_bwd_weights_t),
    INSTANCE(ref_deconvolution_bwd_data_t),
    INSTANCE(ref_deconvolution_fwd_t),
    /* shuffle */
    INSTANCE(ref_shuffle_t<4>), /* f32 or s32 */
    INSTANCE(ref_shuffle_t<1>), /* s8 or u8 */
#endif
    /* eltwise */
    INSTANCE(jit_uni_eltwise_fwd_t<avx512_common, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_eltwise_fwd_t<avx512_common, bf16>),
    INSTANCE(jit_uni_eltwise_bwd_t<avx512_common, f32>),
    INSTANCE(jit_uni_eltwise_bwd_t<avx512_common, bf16>),
#endif
    INSTANCE(jit_uni_eltwise_fwd_t<avx2, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_eltwise_bwd_t<avx2, f32>),
#endif
    INSTANCE(jit_uni_eltwise_fwd_t<sse42, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_eltwise_bwd_t<sse42, f32>),
#endif

    INSTANCE(ref_eltwise_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_eltwise_fwd_t<bf16>),
    INSTANCE(ref_eltwise_bwd_t<f32>),
    INSTANCE(ref_eltwise_bwd_t<bf16>),
#endif
    /* depthwise */
    INSTANCE(jit_uni_depthwise_fwd_t<avx512_common>),
    INSTANCE(jit_uni_depthwise_fwd_t<avx2>),
    INSTANCE(jit_uni_depthwise_fwd_t<sse42>),
    INSTANCE(ref_depthwise_fwd_t<f32>),
    /* eltwise (int) */
    INSTANCE(ref_eltwise_fwd_t<s32>),
    INSTANCE(ref_eltwise_fwd_t<s16>),
    INSTANCE(ref_eltwise_fwd_t<s8>),
    INSTANCE(ref_eltwise_fwd_t<u8>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_eltwise_bwd_t<s32>),
    INSTANCE(ref_eltwise_bwd_t<s16>),
#endif
    /* softmax */
    INSTANCE(jit_uni_softmax_fwd_t<avx512_common>),
    INSTANCE(jit_uni_softmax_fwd_t<avx2>),
    INSTANCE(jit_uni_softmax_fwd_t<sse42>),
    INSTANCE(ref_softmax_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_softmax_bwd_t<f32>),
    /* pool */
    INSTANCE(jit_uni_pooling_fwd_t<avx512_common, bf16>),
    INSTANCE(jit_uni_pooling_bwd_t<avx512_common, bf16>),
#endif
    INSTANCE(jit_uni_pooling_fwd_t<avx512_common, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_pooling_bwd_t<avx512_common, f32>),
#endif
    INSTANCE(jit_uni_pooling_fwd_t<avx, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_pooling_bwd_t<avx, f32>),
#endif
    INSTANCE(jit_uni_pooling_fwd_t<sse42, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_pooling_bwd_t<sse42, f32>),
    INSTANCE(nchw_pooling_fwd_t<bf16>),
    INSTANCE(nchw_pooling_bwd_t<bf16>),
#endif
    INSTANCE(nchw_pooling_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(nchw_pooling_bwd_t<f32>),

    INSTANCE(nhwc_pooling_fwd_t<bf16>),
    INSTANCE(nhwc_pooling_bwd_t<bf16>),
#endif
    INSTANCE(nhwc_pooling_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(nhwc_pooling_bwd_t<f32>),
#endif

    INSTANCE(ref_pooling_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_pooling_fwd_t<bf16, f32>),
    INSTANCE(ref_pooling_bwd_t<f32>),
    INSTANCE(ref_pooling_bwd_t<bf16>),
#endif

    /* pool (int) */
    INSTANCE(jit_uni_i8i8_pooling_fwd_t<avx512_core>),
    INSTANCE(jit_uni_i8i8_pooling_fwd_t<avx2>),
    INSTANCE(jit_sse42_i8i8_pooling_fwd_t),
    INSTANCE(ref_pooling_fwd_t<s32>),
    INSTANCE(ref_pooling_fwd_t<s16, s32>),
    INSTANCE(ref_pooling_fwd_t<s8, s32>),
    INSTANCE(ref_pooling_fwd_t<u8, s32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_pooling_bwd_t<s32>),
    INSTANCE(ref_pooling_bwd_t<s16>),
#endif

    /* lrn */
    INSTANCE(jit_avx512_common_lrn_fwd_t),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_avx512_common_lrn_bwd_t),
#endif
    INSTANCE(jit_uni_lrn_fwd_t<avx2>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_lrn_bwd_t<avx2>),
#endif
    INSTANCE(jit_uni_lrn_fwd_t<sse42>),
    INSTANCE(ref_lrn_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_lrn_bwd_t<f32>),
#endif
    /* batch normalization */
    INSTANCE(jit_uni_batch_normalization_fwd_t<avx512_common, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_batch_normalization_bwd_t<avx512_common, f32>),
    INSTANCE(jit_uni_batch_normalization_fwd_t<avx512_common, bf16>),
    INSTANCE(jit_uni_batch_normalization_bwd_t<avx512_common, bf16>),
#endif
    INSTANCE(jit_uni_batch_normalization_fwd_t<avx2, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_batch_normalization_bwd_t<avx2, f32>),
#endif
    INSTANCE(jit_uni_batch_normalization_fwd_t<sse42, f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(jit_uni_batch_normalization_bwd_t<sse42, f32>),
#endif
    INSTANCE(ncsp_batch_normalization_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ncsp_batch_normalization_bwd_t<f32>),
    INSTANCE(ncsp_batch_normalization_fwd_t<bf16>),
    INSTANCE(ncsp_batch_normalization_bwd_t<bf16>),
#endif
    INSTANCE(nspc_batch_normalization_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(nspc_batch_normalization_bwd_t<f32>),
    INSTANCE(nspc_batch_normalization_fwd_t<bf16>),
    INSTANCE(nspc_batch_normalization_bwd_t<bf16>),
#endif
    INSTANCE(ref_batch_normalization_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_batch_normalization_bwd_t<f32>),
    INSTANCE(ref_batch_normalization_fwd_t<bf16>),
    INSTANCE(ref_batch_normalization_bwd_t<bf16>),
    /* batch normalization (int) */
    INSTANCE(jit_uni_batch_normalization_s8_fwd_t<avx512_core>),
    INSTANCE(jit_uni_batch_normalization_s8_fwd_t<avx2>),
    INSTANCE(ref_batch_normalization_fwd_t<s8>),
#endif
    /* inner product */
    INSTANCE(gemm_inner_product_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(gemm_inner_product_bwd_data_t<f32>),
    INSTANCE(gemm_inner_product_bwd_weights_t<f32>),
#endif
    INSTANCE(ref_inner_product_fwd_t<f32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_inner_product_bwd_data_t<f32, f32, f32, f32>),
    INSTANCE(ref_inner_product_bwd_weights_t<f32>),
    /* inner product (bfloat16) */
    INSTANCE(gemm_bf16_inner_product_fwd_t<f32>),
    INSTANCE(gemm_bf16_inner_product_fwd_t<bf16>),
    INSTANCE(gemm_bf16_inner_product_bwd_data_t<f32>),
    INSTANCE(gemm_bf16_inner_product_bwd_data_t<bf16>),
    INSTANCE(gemm_bf16_inner_product_bwd_weights_t<f32>),
    INSTANCE(gemm_bf16_inner_product_bwd_weights_t<bf16>),
#endif
    /* inner product (int) */
    INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<u8, u8>),
    INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<u8, s8>),
    INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<u8, s32>),
    INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<u8, f32>),
    INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<s8, u8>),
    INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<s8, s8>),
    INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<s8, s32>),
    INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<s8, f32>),
    INSTANCE(ref_inner_product_fwd_t<u8, s8, u8, s32>),
    INSTANCE(ref_inner_product_fwd_t<u8, s8, s8, s32>),
    INSTANCE(ref_inner_product_fwd_t<u8, s8, s32, s32>),
    INSTANCE(ref_inner_product_fwd_t<u8, s8, f32, s32>),
    INSTANCE(ref_inner_product_fwd_t<s16, s16, s32, s32>),
#ifdef ENABLE_UNUSED_PRIM
    INSTANCE(ref_inner_product_bwd_data_t<s32, s16, s16, s32>),
#endif
    /* roi pooling */
    INSTANCE(jit_uni_roi_pooling_fwd_t<avx512_common>),
    INSTANCE(jit_uni_roi_pooling_fwd_t<avx2>),
    INSTANCE(jit_uni_roi_pooling_fwd_t<sse42>),
    INSTANCE(ref_roi_pooling_fwd_t<data_type::f32>),
    /* binary convolution */
    INSTANCE(jit_uni_binary_convolution_fwd_t<avx512_common>),
    INSTANCE(jit_uni_binary_convolution_fwd_t<avx2>),
    INSTANCE(jit_uni_binary_convolution_fwd_t<sse42>),
    INSTANCE(ref_binary_convolution_fwd_t),
    /* binarization */
    INSTANCE(jit_uni_binarization_fwd_t<avx512_common>),
    INSTANCE(jit_uni_binarization_fwd_t<avx2>),
    INSTANCE(jit_uni_binarization_fwd_t<sse42>),
    INSTANCE(ref_binarization_fwd_t<f32>),
    /* deformable convolution */
    INSTANCE(jit_uni_deformable_convolution_fwd_t<avx512_common>),
    INSTANCE(jit_uni_deformable_convolution_fwd_t<avx2>),
    INSTANCE(jit_uni_deformable_convolution_fwd_t<sse42>),
    INSTANCE(ref_deformable_convolution_fwd_t),
    /* eol */
    nullptr,
};
#undef INSTANCE
}

const pd_create_f* cpu_engine_t::get_implementation_list() const {
    return cpu_impl_list;
}

cpu_engine_factory_t engine_factory;

namespace {
// XXX: this is a huge hammer. This disables all and any msan checks on
// primitives outputs.
//
// A proper approach would be an implementation-specific unpoisoning.
void unpoison_outputs(primitive_t *p)
{
    for(auto o: p->outputs()) {
        assert(o->kind() == primitive_kind::memory);
        void *p;
        o->get_data_handle(&p);
        size_t s = ((memory_pd_t *)o->pd())->get_size();
        msan_unpoison(p, s);
    }
}
}

status_t cpu_engine_t::submit(primitive_t *p, event_t *e,
        event_vector &prerequisites) {
    /* FIXME: this should live in primitive execute function... */
    if (mkldnn_verbose()->level) {
        double ms = get_msec();
        p->execute(e);
        ms = get_msec() - ms;
        printf("mkldnn_verbose,exec,%s,%g\n", p->pd()->info(), ms);
        fflush(0);
    } else {
        p->execute(e);
    }
    if (msan_enabled)
        unpoison_outputs(p);
    return success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
