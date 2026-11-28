// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "rms_kernel_base.h"

namespace kernel_selector {
// Feature-axis RMS on b_fs_yx_fsv16 / b_fs_zyx_fsv16 with whole fsv16 blocks (F % 16 == 0, no feature padding).
class RMSKernelFsv16 : public RMSKernelBase {
public:
    using Parent = RMSKernelBase;
    RMSKernelFsv16() : RMSKernelBase("rms_gpu_fsv16") {}
    ~RMSKernelFsv16() override = default;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::ACTIVATION, FusedOpType::QUANTIZE, FusedOpType::ELTWISE, FusedOpType::REORDER};
    }
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const rms_params& params) const override;
    JitConstants GetJitConstants(const rms_params& params, DispatchData dispatchData) const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
};
}  // namespace kernel_selector
