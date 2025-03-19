// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "col_to_im_kernel_base.h"

namespace kernel_selector {
class ColToImKernelOpt : public ColToImKernelBase {
public:
    using Parent = ColToImKernelBase;

    ColToImKernelOpt() : ColToImKernelBase("col_to_im_opt") {}
    virtual ~ColToImKernelOpt() {}

    CommonDispatchData SetDefault(const col_to_im_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const col_to_im_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::REORDER,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
