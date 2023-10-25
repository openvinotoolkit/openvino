// Copyright (C) 2018-2023 Intel Corporation
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
#include "reorder_weights_int4.h"

namespace kernel_selector {

ReorderWeightsKernelSelector::ReorderWeightsKernelSelector() {
    Attach<ReorderWeightsKernel>();
    Attach<ReorderWeightsWinograd2x3Kernel>();
    Attach<ReorderWeightsWinograd6x3Kernel>();
    Attach<ReorderWeightsImage_fyx_b_Kernel>();
    Attach<ReorderWeightsImageWinograd6x3Kernel>();
    Attach<ReorderWeightsBinaryKernel>();
    Attach<ReorderWeightsOpt>();
    Attach<ReorderWeightsKernelInt4>();
}

KernelsData ReorderWeightsKernelSelector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::REORDER);
}
}  // namespace kernel_selector
