// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_kernel_base.h"

namespace kernel_selector {

class FullyConnected_fs_byx_fsv32 : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;
    FullyConnected_fs_byx_fsv32() : Parent("fully_connected_gpu_fs_byx_fsv32") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1, int kernel_number = 0) const override;
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector
