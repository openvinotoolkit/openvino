// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_kernel_selector.h"
#include "gemm_kernel_ref.h"
#include "gemm_kernel_tiled_opt.h"
#include "gemm_kernel_mmad_int8.h"
#include "gemm_kernel_mmad_int8_slm.h"

namespace kernel_selector {
gemm_kernel_selector::gemm_kernel_selector() {
    Attach<GemmKernelRef>();
    Attach<GemmKernelTiledOpt>();
    Attach<GemmKernelMMADint8>();
    Attach<GemmKernelMMADslmInt8>();
}

KernelsData gemm_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::GEMM);
}
}  // namespace kernel_selector
