// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class Convolution_kernel_b_fs_zyx_fsv16_imad : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    using BlockParams = DispatchData::BlockParams;
    Convolution_kernel_b_fs_zyx_fsv16_imad() : ConvolutionKernelBase("convolution_gpu_b_fs_zyx_fsv16_imad") {}
    virtual ~Convolution_kernel_b_fs_zyx_fsv16_imad() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    bool NeedPaddedInput() const override { return true; }
    WeightsLayout GetPreferredWeightsLayout(const convolution_params& p) const override {
        return p.groups > 1 ? WeightsLayout::g_os_is_zyx_osv16_isv16 : WeightsLayout::os_is_zyx_osv16_isv16;
    }

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

    BlockParams GetBlockParams(const convolution_params& params) const;
    float EstimateBlockParamsRatio(const convolution_params& params, const BlockParams& block) const;
    float EstimateRegPressure(const convolution_params& params, const BlockParams& block) const;
    float EstimateOccupancy(const convolution_params& params, const BlockParams& block) const;
    float EstimateSLMUsage(const convolution_params& params, const BlockParams& block) const;
    DispatchData CalcDispatchDataWithBlockParams(const convolution_params& params, const BlockParams& block_params) const;
};
}  // namespace kernel_selector
