// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise_kernel_base.h"

namespace kernel_selector {
class EltwiseKernel_b_fs_yx_fsv16 : public EltwiseKernelBase {
public:
    EltwiseKernel_b_fs_yx_fsv16() : EltwiseKernelBase("eltwise_b_fs_yx_fsv16") {}
    virtual ~EltwiseKernel_b_fs_yx_fsv16() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::QUANTIZE,
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE
        };
    }

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants MakeLoadJitConstants(const eltwise_params& params, bool useVload8) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
    DispatchData SetDefault(const eltwise_params& params) const override;
};
}  // namespace kernel_selector
