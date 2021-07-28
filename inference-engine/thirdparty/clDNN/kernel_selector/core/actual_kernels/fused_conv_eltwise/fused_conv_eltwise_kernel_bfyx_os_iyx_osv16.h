// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fused_conv_eltwise_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class fused_conv_eltwise_kernel_bfyx_os_iyx_osv16 : public fused_conv_eltwise_kernel_base {
public:
    using Parent = fused_conv_eltwise_kernel_base;
    fused_conv_eltwise_kernel_bfyx_os_iyx_osv16();
    virtual ~fused_conv_eltwise_kernel_bfyx_os_iyx_osv16() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferreddWeightsLayout(const fused_conv_eltwise_params &) const override;
    JitConstants GetJitConstants(const fused_conv_eltwise_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    bool NeedPaddedInput() const override { return true; }
    DispatchData SetDefault(const fused_conv_eltwise_params& arg, int autoTuneIndex = -1) const override;

private:
    struct AutoTuneOption {
        size_t blockWidth;
        size_t blockHeight;
        size_t prefetch;
        std::string exeMode;
    };

    AutoTuneOption GetAutoTuneOptions(const Params& arg, int autoTuneIndex) const;

    std::vector<AutoTuneOption> autoTuneOptions = {};
};
}  // namespace kernel_selector
