// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {
class MVNKernel_b_fs_yx_fsv16_imad : public MVNKernelBase {
public:
    using Parent = MVNKernelBase;
    MVNKernel_b_fs_yx_fsv16_imad() : MVNKernelBase("mvn_gpu_b_fs_yx_fsv16_imad") {}
    virtual ~MVNKernel_b_fs_yx_fsv16_imad() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    struct MultiDispatchData {
        DispatchData stage_1;
        DispatchData stage_2;
        DispatchData stage_final;

        size_t item_groups;
    };

    bool Validate(const Params&, const optional_params&) const override;
    DispatchData SetDefault(const mvn_params& params) const override;
    JitConstants GetJitConstants(const mvn_params& params, DispatchData dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE
        };
    }

    KernelsData GetMultiStageKernelsData(const mvn_params& params, const optional_params&) const;
    MultiDispatchData SetDefaultForMulti(const mvn_params& params) const;

private:
    Datatype GetAccumulatorType(const mvn_params& params) const;
};
}  // namespace kernel_selector
