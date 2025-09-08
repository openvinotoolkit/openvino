// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "region_yolo_kernel_selector.h"
#include "region_yolo_kernel_ref.h"

namespace kernel_selector {

region_yolo_kernel_selector::region_yolo_kernel_selector() { Attach<RegionYoloKernelRef>(); }

KernelsData region_yolo_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::REGION_YOLO);
}
}  // namespace kernel_selector
