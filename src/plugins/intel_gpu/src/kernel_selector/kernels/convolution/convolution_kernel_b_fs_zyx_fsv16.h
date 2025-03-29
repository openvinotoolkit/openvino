// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_b_fs_zyx_fsv16 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;

    explicit ConvolutionKernel_b_fs_zyx_fsv16(Datatype use_data_type) :
        ConvolutionKernelBase(use_data_type == Datatype::F32 ? "gen9_common_conv_fwd_data_f32" : "gen9_common_conv_fwd_data_f16"),
        use_data_type(use_data_type) {}

    virtual ~ConvolutionKernel_b_fs_zyx_fsv16() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params& params) const override {
        bool is_3d_case = params.inputs[0].GetLayout() != DataLayout::bs_fs_yx_bsv16_fsv16;
        if (params.inputs[0].Feature().v == 3 && params.inputs[0].GetLayout() == DataLayout::bfzyx) {
            return WeightsLayout::os_zyxi_osv16;
        } else if (use_data_type == Datatype::F32 && params.inputs[0].Batch().v % 16 == 0) {
            if (is_3d_case)
                return (params.groups > 1) ? WeightsLayout::g_is_os_zyx_isv16_osv16 : WeightsLayout::is_os_zyx_isv16_osv16;
            else
                return (params.groups > 1) ? WeightsLayout::g_is_os_yx_isv16_osv16 : WeightsLayout::is_os_yx_isv16_osv16;
        } else if (use_data_type == Datatype::F16 && params.inputs[0].Batch().v % 32 == 0) {
            if (is_3d_case)
                return (params.groups > 1) ? WeightsLayout::g_os_is_zyx_isv8_osv16_isv2 : WeightsLayout::os_is_zyx_isv8_osv16_isv2;
            else
                return (params.groups > 1) ? WeightsLayout::g_os_is_yx_isv8_osv16_isv2 : WeightsLayout::os_is_yx_isv8_osv16_isv2;
        } else {
            return (params.groups > 1) ? WeightsLayout::g_os_is_zyx_isv16_osv16 : WeightsLayout::os_is_zyx_isv16_osv16;
        }
    }
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

    // This class is base one for FP16 and FP32 classes
    Datatype use_data_type;
};
}  // namespace kernel_selector
