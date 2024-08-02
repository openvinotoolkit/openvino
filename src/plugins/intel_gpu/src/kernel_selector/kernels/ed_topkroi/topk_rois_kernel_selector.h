// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

/**
 * GPU kernel selector for the ExperimentalDetectronTopKROIS-6 operation
 */
class experimental_detectron_topk_rois_kernel_selector : public kernel_selector_base {
public:
    static experimental_detectron_topk_rois_kernel_selector &Instance();

    experimental_detectron_topk_rois_kernel_selector();

    KernelsData GetBestKernels(const Params &params) const override;
};
}  // namespace kernel_selector
