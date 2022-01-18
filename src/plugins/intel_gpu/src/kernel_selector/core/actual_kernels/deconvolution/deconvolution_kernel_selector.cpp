// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_kernel_selector.h"
#include "deconvolution_kernel_ref.h"
#include "deconvolution_kernel_bfyx_opt.h"
#include "deconvolution_kernel_b_fs_zyx_fsv16.h"
#include "deconvolution_kernel_b_fs_zyx_fsv16_dw.h"
#include "deconvolution_kernel_imad_ref.hpp"
#include "deconvolution_kernel_imad_along_f_tile_bfx.hpp"

namespace kernel_selector {
deconvolution_kernel_selector::deconvolution_kernel_selector() {
    Attach<DeconvolutionKernelRef>();
    Attach<DeconvolutionKernel_bfyx_opt>();
    Attach<DeconvolutionKernel_b_fs_zyx_fsv16>();
    Attach<DeconvolutionKernel_b_fs_zyx_fsv16_dw>();
    Attach<DeconvolutionKernel_imad_ref>();
    Attach<DeconvolutionKernel_imad_along_f_tile_bfx>();
}

KernelsData deconvolution_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::DECONVOLUTION);
}
}  // namespace kernel_selector
