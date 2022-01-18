// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include "convolution_kernel_bfyx_to_b_fs_yx_fsv16.h"
#include <string>
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16 : public ConvolutionKernel_bfyx_to_bfyx_f16 {
public:
    ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16();
    virtual ~ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16() {}

    ParamsKey GetSupportedKey() const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override {
        return WeightsLayout::os_is_yx_isv16_osv16;
    }
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

    bool Validate(const Params& p, const optional_params& o) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector
