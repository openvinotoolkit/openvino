// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "kernel_selector.h"

namespace kernel_selector {

/*
 * CTCLoss kernel selector.
 */
class ctc_loss_kernel_selector : public kernel_selector_base {
public:
    ctc_loss_kernel_selector();
    KernelsData GetBestKernels(const Params& params) const override;
    static ctc_loss_kernel_selector& Instance();
};

}  // namespace kernel_selector
