// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_selector.h"
#include "convolution_kernel_ref.h"
#include "convolution_kernel_bfyx_1x1_opt.h"
#include "convolution_kernel_bfyx_gemm_like.h"
#include "convolution_kernel_bfyx_direct_10_12_16.h"
#include "convolution_kernel_bfyx_os_iyx_osv16.h"
#include "convolution_kernel_bfyx_os_iyx_osv32.h"
#include "convolution_kernel_bfyx_iyxo.h"
#include "convolution_kernel_yxfb_ref.h"
#include "convolution_kernel_yxfb_yxio_b16.h"
#include "convolution_kernel_yxfb_yxio_b8.h"
#include "convolution_kernel_yxfb_yxio_b1_block_multiple_x.h"
#include "convolution_kernel_winograd_2x3_s1.h"
#include "convolution_kernel_bfyx_1x1.h"
#include "convolution_kernel_bfyx_1x1_gemm_buf.h"
#include "convolution_kernel_winograd_2x3_s1_fused.h"
#include "convolution_kernel_winograd_6x3_s1_fused.h"
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
#include "deformable_convolution_kernel_bfyx_opt.h"
#include "convolution_kernel_b_fs_zyx_fsv16_fp32.h"
#include "convolution_kernel_b_fs_zyx_fsv16_fp16.h"
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
    Attach<ConvolutionKernel_bfyx_os_iyx_osv32>();
    Attach<ConvolutionKernel_bfyx_iyxo>();
    Attach<ConvolutionKernel_bfyx_1x1>();
    Attach<ConvolutionKernel_bfyx_1x1_gemm_buf>();
    Attach<ConvolutionKernel_bfyx_depthwise_weights_lwg>();

    // yxfb fp
    Attach<ConvolutionKernel_yxfb_Ref>();
    Attach<ConvolutionKernel_yxfb_yxio_b16>();
    Attach<ConvolutionKernel_yxfb_yxio_b8>();
    Attach<ConvolutionKernel_yxfb_yxio_b1_block_multiple_x>();

    // Winograd
    Attach<ConvolutionKernel_Winograd_2x3_s1>();
    Attach<ConvolutionKernel_Winograd_2x3_s1_fused>();
    Attach<ConvolutionKernel_Winograd_6x3_s1_fused>();

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

    Attach<DeformableConvolutionKernel_bfyx_Ref>();
    Attach<DeformableConvolutionKernel_bfyx_opt>();
}

KernelsData convolution_kernel_selector::GetBestKernels(const Params& params) const {
    return GetAutoTuneBestKernel(params, KernelType::CONVOLUTION);
}

}  // namespace kernel_selector
