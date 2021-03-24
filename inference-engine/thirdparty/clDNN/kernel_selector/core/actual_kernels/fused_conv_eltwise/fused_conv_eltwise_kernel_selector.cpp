// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_conv_eltwise_kernel_selector.h"
#include "fused_conv_eltwise_kernel_bfyx_1x1_opt.h"
#include "fused_conv_eltwise_kernel_bfyx_os_iyx_osv16.h"
#include "fused_conv_eltwise_kernel_yxfb_yxio_b16.h"
#include "fused_conv_eltwise_kernel_bfyx_iyxo.h"

namespace kernel_selector {
fused_conv_eltwise_kernel_selector::fused_conv_eltwise_kernel_selector() {
    Attach<fused_conv_eltwise_kernel_yxfb_yxio_b16>();
    Attach<fused_conv_eltwise_kernel_bfyx_1x1_opt>();
    Attach<fused_conv_eltwise_kernel_bfyx_os_iyx_osv16>();
    Attach<fused_conv_eltwise_kernel_bfyx_iyxo>();
}

KernelsData fused_conv_eltwise_kernel_selector::GetBestKernels(const Params& params,
                                                               const optional_params& options) const {
    return GetAutoTuneBestKernel(params, options, KernelType::FUSED_CONV_ELTWISE);
}
}  // namespace kernel_selector
