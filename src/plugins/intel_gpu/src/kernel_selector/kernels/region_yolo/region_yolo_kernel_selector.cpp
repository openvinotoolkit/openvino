// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "region_yolo_kernel_selector.h"
#include "region_yolo_kernel_ref.h"

namespace kernel_selector {

region_yolo_kernel_selector::region_yolo_kernel_selector() { Attach<RegionYoloKernelRef>(); }

KernelsData region_yolo_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::REGION_YOLO);
}
}  // namespace kernel_selector
