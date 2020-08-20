/*
// Copyright (c) 2016-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "convolution_kernel_selector.h"
#include "convolution_kernel_ref.h"
#include "convolution_kernel_bfyx_1x1_opt.h"
#include "convolution_kernel_bfyx_gemm_like.h"
#include "convolution_kernel_bfyx_direct_10_12_16.h"
#include "convolution_kernel_bfyx_os_iyx_osv16.h"
#include "convolution_kernel_bfyx_os_iyx_osv16_2_sg.h"
#include "convolution_kernel_bfyx_iyxo.h"
#include "convolution_kernel_yxfb_ref.h"
#include "convolution_kernel_yxfb_yxio_b16.h"
#include "convolution_kernel_yxfb_yxio_b8.h"
#include "convolution_kernel_yxfb_yxio_b1_block.h"
#include "convolution_kernel_yxfb_yxio_b1_block_multiple_x.h"
// #include "convolution_kernel_bfyx_3x3_dw_opt.h"
#include "convolution_kernel_winograd_2x3_s1.h"
#include "convolution_kernel_bfyx_1x1.h"
#include "convolution_kernel_bfyx_1x1_gemm_buf.h"
#include "convolution_kernel_winograd_2x3_s1_fused.h"
#include "convolution_kernel_winograd_6x3_s1_fused.h"
#include "convolution_kernel_mmad.h"
#include "convolution_kernel_mmad_blocks.h"
#include "convolution_kernel_imad_byxf_af32_depthwise.h"
#include "convolution_kernel_bfyx_depthwise_weights_lwg.h"
#include "convolution_kernel_imad.h"
#include "convolution_kernel_fs_byx_fsv32.h"
#include "convolution_kernel_fs_byx_fsv32_1x1.h"
#include "convolution_kernel_bfyx_to_fs_byx_fsv32.h"
#include "convolution_kernel_fs_byx_fsv32_depthwise.h"
#include "convolution_kernel_b_fs_yx_fsv16_depthwise.h"
#include "convolution_kernel_b_fs_yx_fsv16_1x1.h"
#include "convolution_kernel_b_fs_yx_fsv16.h"
#include "convolution_kernel_bfyx_to_b_fs_yx_fsv16.h"
#include "deformable_convolution_kernel_bfyx_ref.h"
#include "deformable_convolution_kernel_bfyx_conv.h"
#include "deformable_convolution_kernel_bfyx_interp.h"
#include "convolution_kernel_b_fs_zyx_fsv16_fp32.h"
#include "convolution_kernel_b_fs_zyx_fsv16_fp16.h"
#include "convolution_kernel_imad_byxf_af32_1x1.h"
#include "convolution_kernel_imad_b_fs_yx_fsv4_1x1.h"
#include "convolution_kernel_imad_b_fs_yx_fsv4_dw.hpp"
#include "convolution_kernel_mmad_bfyx_to_b_fs_yx_fsv4.h"
#include "convolution_kernel_mmad_b_fs_yx_fsv32.h"
#include "convolution_kernel_mmad_b_fs_yx_fsv32_dw.h"
#include "convolution_kernel_mmad_bfyx_to_b_fs_yx_fsv32.h"
#include "convolution_kernel_bfyx_to_bs_fs_yx_bsv16_fsv16.h"
#include "convolution_kernel_b_fs_yx_fsv16_imad_1x1.h"
#include "convolution_kernel_b_fs_zyx_fsv16_imad.h"
#include "convolution_kernel_b_fs_yx_fsv_16_32_imad_dw.hpp"
#include "convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_1x1.h"
#include "convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3.h"
#include "convolution_kernel_b_fs_yx_fsv4_int8.h"

namespace kernel_selector {
convolution_kernel_selector::convolution_kernel_selector() {
    Attach<ConvolutionKernel_Ref>();
    Attach<DeformableConvolutionKernel_bfyx_Ref>();

    // b_fs_yx_fsv16 and b_fs_zyx_fsv16 int8
    Attach<Convolution_kernel_b_fs_yx_fsv16_imad_1x1>();
    Attach<Convolution_kernel_b_fs_zyx_fsv16_imad>();

    // b_fs_yx_fsv16 and b_fs_zyx_fsv16
    Attach<ConvolutionKernel_b_fs_yx_fsv16_depthwise>();
    Attach<ConvolutionKernel_b_fs_yx_fsv16_1x1>();
    Attach<ConvolutionKernel_b_fs_yx_fsv16>();
    Attach<ConvolutionKernel_bfyx_to_bfyx_f16>();
    Attach<ConvolutionKernel_b_fs_zyx_fsv16_fp32>();
    Attach<ConvolutionKernel_b_fs_zyx_fsv16_fp16>();

    // bs_fs_yx_bsv16_fsv16
    Attach<ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16>();
    Attach<Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_1x1>();
    Attach<Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3>();

    // fs_byx_fsv32
    Attach<ConvolutionKernel_fs_byx_fsv32>();
    Attach<ConvolutionKernel_fs_byx_fsv32_1x1>();
    Attach<ConvolutionKernel_fs_byx_fsv32_depthwise>();
    Attach<ConvolutionKernel_bfyx_to_fs_byx_fsv32>();

    // bfyx fp
    Attach<convolution_kernel_bfyx_1x1_opt>();
    Attach<ConvolutionKernel_bfyx_GEMMLike>();
    Attach<ConvolutionKernel_bfyx_Direct_10_10_12>();
    Attach<ConvolutionKernel_bfyx_os_iyx_osv16>();
    Attach<ConvolutionKernel_bfyx_iyxo>();
    Attach<ConvolutionKernel_bfyx_1x1>();
    Attach<ConvolutionKernel_bfyx_1x1_gemm_buf>();
    Attach<ConvolutionKernel_bfyx_depthwise_weights_lwg>();
    // commented out to not get in our way, will enable in future after autotuning
    // Attach<ConvolutionKernel_bfyx_os_iyx_osv16_2_sg>();

    // yxfb fp
    Attach<ConvolutionKernel_yxfb_Ref>();
    Attach<ConvolutionKernel_yxfb_yxio_b16>();
    Attach<ConvolutionKernel_yxfb_yxio_b8>();
    Attach<ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x>();
    // Attach<ConvolutionKernel_yxfb_yxio_b1_block>(); // TODO: need to finish integration
    // Attach<ConvolutionKernel_bfyx_3x3_dw_opt>();

    // Winograd
    Attach<ConvolutionKernel_Winograd_2x3_s1>();
    Attach<ConvolutionKernel_Winograd_2x3_s1_fused>();
    Attach<ConvolutionKernel_Winograd_6x3_s1_fused>();

    // byxf_af32 int8
    Attach<ConvolutionKernel_mmad>();
    Attach<ConvolutionKernel_mmad_blocks>();
    Attach<ConvolutionKernel_imad_byxf_af32_1x1>();
    Attach<ConvolutionKernel_imad_byxf_af32_depthiwise>();

    // b_fs_yx_fsv4 kernels
    Attach<ConvolutionKernel_imad>();
    Attach<ConvolutionKernel_imad_b_fs_yx_fsv4_1x1>();
    Attach<ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4>();
    Attach<ConvolutionKernel_imad_b_fs_yx_fsv4_dw>();
    Attach<ConvolutionKernel_b_fs_yx_fsv4_int8>();

    // b_fs_yx_fsv32 kernels
    Attach<ConvolutionKernel_mmad_b_fs_yx_fsv32>();
    Attach<ConvolutionKernel_mmad_b_fs_yx_fsv32_dw>();
    Attach<ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32>();
    Attach<ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw>();
}

KernelsData convolution_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetAutoTuneBestKernel(params, options, KernelType::CONVOLUTION);
}

deformable_conv_kernel_selector::deformable_conv_kernel_selector() {
    Attach<DeformableConvolutionKernel_bfyx_conv>();
}

KernelsData deformable_conv_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetAutoTuneBestKernel(params, options, KernelType::CONVOLUTION);
}

deformable_interp_kernel_selector::deformable_interp_kernel_selector() {
    Attach<DeformableConvolutionKernel_bfyx_interp>();
}

KernelsData deformable_interp_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetAutoTuneBestKernel(params, options, KernelType::CONVOLUTION);
}


}  // namespace kernel_selector
