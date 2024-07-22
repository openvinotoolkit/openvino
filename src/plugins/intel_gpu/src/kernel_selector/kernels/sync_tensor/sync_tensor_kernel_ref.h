// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sync_tensor_kernel_base.h"

namespace kernel_selector {
class SyncTensorKernelRef : public SyncTensorKernelBase {
public:
    using Parent = SyncTensorKernelBase;
    SyncTensorKernelRef() : Parent("sync_tensor_ref") {}
    virtual ~SyncTensorKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    DispatchData SetDefault(const sync_tensor_params& params) const override;
    JitConstants GetJitConstants(const sync_tensor_params& params, DispatchData dispatchData) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
