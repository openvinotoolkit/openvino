// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "kernel_base_opencl.h"

namespace kernel_selector {

/**
 * CTCLoss reference kernel parameters.
 */
struct ctc_loss_params : base_params {
    ctc_loss_params() : base_params(KernelType::CTC_LOSS) {}
    bool preprocess_collapse_repeated = false;
    bool ctc_merge_repeated = true;
    bool unique = false;
};

/**
 * Reference kernel for CTCLoss.
 */
class CTCLossKernelRef : public KernelBaseOpenCL {
public:
    CTCLossKernelRef() : KernelBaseOpenCL{"ctc_loss_ref"} {}

private:
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const ctc_loss_params& kernel_params) const;
};

}  // namespace kernel_selector
