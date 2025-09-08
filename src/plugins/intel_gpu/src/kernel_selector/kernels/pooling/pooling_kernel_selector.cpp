// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_kernel_selector.h"
#include "pooling_kernel_gpu_ref.h"
#include "pooling_kernel_gpu_byxf_opt.h"
#include "pooling_kernel_gpu_bfyx_block_opt.h"
#include "pooling_kernel_gpu_byxf_padding_opt.h"
#include "pooling_kernel_gpu_int8_ref.h"
#include "pooling_kernel_gpu_b_fs_yx_fsv4.h"
#include "pooling_kernel_gpu_fs_b_yx_fsv32.h"
#include "pooling_kernel_gpu_b_fs_yx_fsv16.h"
#include "pooling_kernel_gpu_bsv16_fsv16.h"
#include "pooling_kernel_gpu_b_fs_zyx_fsv16_imad.h"
#include "pooling_kernel_gpu_bs_fs_yx_bsv16_fsv16.h"

namespace kernel_selector {

pooling_kernel_selector::pooling_kernel_selector() {
    Attach<PoolingKernelGPURef>();
    Attach<PoolingKernelGPUByxfOpt>();
    Attach<PoolingKernelGPUBfyxBlockOpt>();
    Attach<PoolingKernelGPUByxfPaddingOpt>();
    Attach<PoolingKernelGPUInt8Ref>();
    Attach<PoolingKerneGPU_b_fs_yx_fsv4>();
    Attach<PoolingKerneGPU_fs_b_yx_fsv32>();
    Attach<PoolingKernel_b_fs_yx_fsv16>();
    Attach<PoolingKernel_bsv16_fsv16>();
    Attach<PoolingKernelGPU_b_fs_zyx_fsv16_imad>();
    Attach<Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16>();
}

KernelsData pooling_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::POOLING);
}
}  // namespace kernel_selector
