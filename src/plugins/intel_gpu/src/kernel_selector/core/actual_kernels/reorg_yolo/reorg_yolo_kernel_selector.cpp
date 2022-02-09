// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorg_yolo_kernel_selector.h"
#include "reorg_yolo_kernel_ref.h"

namespace kernel_selector {

reorg_yolo_kernel_selector::reorg_yolo_kernel_selector() { Attach<ReorgYoloKernelRef>(); }

KernelsData reorg_yolo_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::REORG_YOLO);
}
}  // namespace kernel_selector
