// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multiclass_nms_kernel_selector.h"

#include "multiclass_nms_kernel_ref.h"

namespace kernel_selector {
multiclass_nms_kernel_selector::multiclass_nms_kernel_selector() {
    Attach<MulticlassNmsKernelRef>();
}

multiclass_nms_kernel_selector&
multiclass_nms_kernel_selector::Instance() {
    static multiclass_nms_kernel_selector instance_;
    return instance_;
}

KernelsData multiclass_nms_kernel_selector::GetBestKernels(
    const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::MULTICLASS_NMS);
}
}  // namespace kernel_selector
