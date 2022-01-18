// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_kernel_selector.h"
#include "reorder_weights_kernel.h"
#include "reorder_weights_winograd_2x3_kernel.h"
#include "reorder_weights_winograd_6x3_kernel.h"
#include "reorder_weights_image_fyx_b_kernel.h"
#include "reorder_weights_image_winograd_6x3_kernel.h"
#include "reorder_weights_binary_kernel.h"
#include "reorder_weights_opt.h"

namespace kernel_selector {

ReorderWeightsKernelSelctor::ReorderWeightsKernelSelctor() {
    Attach<ReorderWeightsKernel>();
    Attach<ReorderWeightsWinograd2x3Kernel>();
    Attach<ReorderWeightsWinograd6x3Kernel>();
    Attach<ReorderWeightsImage_fyx_b_Kernel>();
    Attach<ReorderWeightsImageWinograd6x3Kernel>();
    Attach<ReorderWeightsBinaryKernel>();
    Attach<ReorderWeightsOpt>();
}

KernelsData ReorderWeightsKernelSelctor::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::REORDER);
}
}  // namespace kernel_selector