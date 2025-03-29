// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorg_yolo_kernel_selector.h"
#include "reorg_yolo_kernel_ref.h"

namespace kernel_selector {

reorg_yolo_kernel_selector::reorg_yolo_kernel_selector() { Attach<ReorgYoloKernelRef>(); }

KernelsData reorg_yolo_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::REORG_YOLO);
}
}  // namespace kernel_selector
