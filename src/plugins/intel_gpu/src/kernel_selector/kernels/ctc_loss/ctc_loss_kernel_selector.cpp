// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ctc_loss_kernel_selector.hpp"

#include "ctc_loss_kernel_ref.hpp"

namespace kernel_selector {

ctc_loss_kernel_selector::ctc_loss_kernel_selector() {
    Attach<CTCLossKernelRef>();
}

KernelsData ctc_loss_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::CTC_LOSS);
}

ctc_loss_kernel_selector& ctc_loss_kernel_selector::Instance() {
    static ctc_loss_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
