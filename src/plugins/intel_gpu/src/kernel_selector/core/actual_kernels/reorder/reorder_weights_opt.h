// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderWeightsOpt : public ReorderKernelBase {
public:
    ReorderWeightsOpt() : ReorderKernelBase("reorder_weights_opt") {}
    virtual ~ReorderWeightsOpt() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const reorder_weights_params& arg) const override;

protected:
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const reorder_weights_params& params) const override;
};
}  // namespace kernel_selector
