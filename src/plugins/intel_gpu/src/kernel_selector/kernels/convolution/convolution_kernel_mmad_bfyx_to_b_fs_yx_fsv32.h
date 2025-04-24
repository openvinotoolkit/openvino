// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32() : ConvolutionKernelBase("convolution_gpu_mmad_bfyx_to_b_fs_yx_fsv32") {}
    virtual ~ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &p) const override {
        if (p.outputs[0].GetDType() == Datatype::F16 || p.outputs[0].GetDType() == Datatype::F32 ||
            p.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 || p.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv16) {
            if (p.outputs[0].Dimentions() == 5) {
                return WeightsLayout::os_is_zyx_osv32_isv4;
            } else {
                return WeightsLayout::os_is_yx_osv32_isv4;
            }
        } else {
            return WeightsLayout::os_is_yx_osv32_isv4_swizzled_by_2;
        }
    }
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

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
