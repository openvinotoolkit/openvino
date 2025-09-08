// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "fully_connected_kernel_base.h"

namespace kernel_selector {

class FullyConnected_GEMV : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;

    FullyConnected_GEMV() : Parent("fully_connected_gpu_gemv") {}

    using FullyConnectedKernelBase::GetTunedKernelsDataByIndex;
    KernelsData GetTunedKernelsDataByIndex(const Params& params, const int autoTuneIndex = -1) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    DispatchData SetDefault(const fully_connected_params& params,
                            int autoTuneIndex = -1,
                            int kernel_number = 0) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::ACTIVATION, FusedOpType::ELTWISE, FusedOpType::SWIGLU};
    }
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;
};
}  // namespace kernel_selector