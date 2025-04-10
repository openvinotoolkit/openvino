// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_kernel_base.h"

namespace kernel_selector {

class FullyConnected_bs_f_bsv16_b1 : public FullyConnectedKernelBase {
public:
    FullyConnected_bs_f_bsv16_b1() : FullyConnectedKernelBase("fully_connected_gpu_bs_f_bsv16_b1") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    JitConstants GetJitConstants(const fully_connected_params& params,
                                 const FullyConnectedKernelBase::DispatchData& dispatchData) const override;
    DispatchData SetDefault(const fully_connected_params& arg, int autoTuneIndex = -1, int kernel_number = 0) const override;
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector
