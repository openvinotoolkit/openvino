// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "convolution_kernel_base.h"
#include <vector>
#include <string>

namespace kernel_selector {
class ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw();
    virtual ~ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw() {}

    ParamsKey GetSupportedKey() const override;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsData GetKernelsDataForAutoTune(const Params & params, const optional_params & options) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params & params, const optional_params & options, int autoTuneIndex = -1) const override;

protected:
    bool Validate(const Params& params, const optional_params& options) const override;
    WeightsLayout GetPreferredWeightsLayout(const convolution_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

    bool NeedPaddedInput() const override { return false; }
    bool HasPaddedInput(const convolution_params& params) const;
    bool ParamsHavePadding(const convolution_params& params) const;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;

    struct AutoTuneParams {
        size_t simd;
        size_t tile_x;
        size_t lws0;
        size_t lws1;
        bool preload_input_slm;
        std::string exeMode;
    };
    std::vector<AutoTuneParams> all_tune_params;

    AutoTuneParams GetAutoTuneParams(const convolution_params& params, int index) const;
    bool ValidateAutoTuneParams(const convolution_params& params, const AutoTuneParams& tparams) const;
};
}  // namespace kernel_selector
