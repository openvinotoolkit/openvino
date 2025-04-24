// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {
class MVNKernel_bs_fs_yx_bsv32 : public MVNKernelBase {
public:
    using Parent = MVNKernelBase;
    MVNKernel_bs_fs_yx_bsv32() : MVNKernelBase("mvn_gpu_b_fs_yx_fsv16_imad") {}
    virtual ~MVNKernel_bs_fs_yx_bsv32() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    struct MultiDispatchData {
        DispatchData stage_1;
        DispatchData stage_2;
        DispatchData stage_final;

        size_t item_groups;
    };

    bool Validate(const Params&) const override;
    JitConstants GetJitConstants(const mvn_params& params, DispatchData dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE
        };
    }

    KernelsData GetMultiStageKernelsData(const mvn_params& params, bool) const;
    MultiDispatchData SetDefaultForMulti(const mvn_params& params, bool) const;

private:
    Datatype GetAccumulatorType(const mvn_params& params) const;
    std::vector<size_t> GetFinalKernelLws(const std::vector<size_t>& gws, uint64_t max_wg) const;
};
}  // namespace kernel_selector
