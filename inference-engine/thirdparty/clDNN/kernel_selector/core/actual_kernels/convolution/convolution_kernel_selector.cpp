/*
// Copyright (c) 2016-2018 Intel Corporation
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
#include "convolution_kernel_bfyx_ref.h"
#include "convolution_kernel_bfyx_gemm_like.h"
#include "convolution_kernel_bfyx_direct_10_12_16.h"
#include "convolution_kernel_bfyx_os_iyx_osv16.h"
#include "convolution_kernel_yxfb_ref.h"
#include "convolution_kernel_yxfb_yxio_b16.h"
#include "convolution_kernel_yxfb_yxio_b8.h"
#include "convolution_kernel_yxfb_yxio_b1_block.h"
#include "convolution_kernel_yxfb_yxio_b1_block_multiple_x.h"
#include "convolution_kernel_tutorial.h"
#include "convolution_kernel_bfyx_3x3_dw_opt.h"
#include "convolution_kernel_winograd_2x3_s1.h"
#include "convolution_kernel_bfyx_1x1.h"
#include "convolution_kernel_bfyx_1x1_gemm_buf.h"
#include "convolution_kernel_winograd_2x3_s1_fused.h"
#include "convolution_kernel_winograd_6x3_s1_fused.h"
#include "convolution_kernel_MMAD.h"
#include "convolution_kernel_MMAD_blocks.h"
#include "convolution_kernel_1x1_gemm_MMAD.h"
#include "convolution_kernel_byxf_af32_depthwise.h"
#include "convolution_kernel_mmad_batched.h"
#include "convolution_kernel_bfyx_depthwise_weights_lwg.h"

#include <iostream>
 
namespace kernel_selector 
{
    convolution_kernel_selector::convolution_kernel_selector()
    {
        Attach<ConvolutionKernel_bfyx_Ref>();
        Attach<ConvolutionKernel_bfyx_GEMMLike>();
        Attach<ConvolutionKernel_bfyx_Direct_10_10_12>();
        Attach<ConvolutionKernel_bfyx_os_iyx_osv16>();
        Attach<ConvolutionKernel_yxfb_Ref>();
        Attach<ConvolutionKernel_yxfb_yxio_b16>();
        Attach<ConvolutionKernel_yxfb_yxio_b8>();
        //Attach<ConvolutionKernel_yxfb_yxio_b1_block>(); // TODO: need to finish integration
        Attach<ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x>();
        Attach<ConvolutionKernel_bfyx_3x3_dw_opt>();
        Attach<ConvolutionKernel_Winograd_2x3_s1>();
        Attach<ConvolutionKernel_Winograd_2x3_s1_fused>();
        Attach<ConvolutionKernel_Winograd_6x3_s1_fused>();
        Attach<ConvolutionKernel_bfyx_1x1>();
        Attach<ConvolutionKernel_bfyx_1x1_gemm_buf>();
        Attach<ConvolutionKernel_MMAD>();
        Attach<ConvolutionKernel_MMAD_blocks>();
        Attach<ConvolutionKernel_1x1_gemm_MMAD>();
        Attach<ConvolutionKernel_byxf_af32_depthiwise>();
        Attach<ConvolutionKernel_mmad_batched>();
        Attach<ConvolutionKernel_bfyx_depthwise_weights_lwg>();
        //Attach<ConvolutionKernel_Tutorial>(); //In order to use this implementation for tutorial purposes please uncomment this line
    }

    KernelsData convolution_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const
    {
        //const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        //std::cout << orgParams.to_string() << std::endl;
        return GetAutoTuneBestKernel(params, options, KernelType::CONVOLUTION);
    }
}