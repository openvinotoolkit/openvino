/*
// Copyright (c) 2016 Intel Corporation
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

#include "pooling_kernel_selector.h"
#include "pooling_kernel_gpu_ref.h"
#include "pooling_kernel_gpu_byxf_opt.h"
#include "pooling_kernel_gpu_average_opt.h"
#include "pooling_kernel_gpu_bfyx_block_opt.h"
#include "pooling_kernel_gpu_byxf_padding_opt.h"
#include "pooling_kernel_gpu_byxf_af32.h"
#include "pooling_kernel_gpu_int8_ref.h"
#include "pooling_kernel_gpu_fs_bs_yx_bsv4_fsv32.h"

namespace kernel_selector {

    pooling_kernel_selector::pooling_kernel_selector()
    {
        Attach<PoolingKernelGPURef>();
        //Attach<PoolingKernelGPUAverageOpt>(); TODO: fix the kernel as it reads out of bounds now
        Attach<PoolingKernelGPUByxfOpt>();
        Attach<PoolingKernelGPUBfyxBlockOpt>();
        Attach<PoolingKernelGPUByxfPaddingOpt>();
        Attach<PoolingKernelGPUInt8Ref>();
        Attach<PoolingKerneGPU_byxf_af32>();
        Attach<PoolingKerneGPU_fs_bs_yx_bsv4_fsv32>();
    }

    KernelsData pooling_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const
    {
        return GetNaiveBestKernel(params, options, KernelType::POOLING);
    }
}