/*
// Copyright (c) 2018-2019 Intel Corporation
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
*/

#pragma once

#include "convolution_kernel_base.h"
#include <vector>
#include <string>

namespace kernel_selector {

class ConvolutionKernel_imad_b_fs_yx_fsv4_dw : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_imad_b_fs_yx_fsv4_dw();

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& kd) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    bool NeedPaddedInput() const override { return false; }

    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override {
        return WeightsLayout::gs_oi_yxs_gsv4_yxsv4;
    }

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

    struct AutoTuneParams {
        size_t block_x;
        size_t block_y;
        size_t tiled_simd;
        bool tiled;
        bool preload_input;
        bool preload_weights;

        std::string exeMode;
    };

    std::vector<AutoTuneParams> all_tune_params;

    bool ValidateAutoTuneParams(const convolution_params& params, const AutoTuneParams& tune_params) const;
    AutoTuneParams GetAutoTuneParams(const convolution_params& params, int index) const;
    KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                           const optional_params& options,
                                           int autoTuneIndex = -1) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params,
                                          const optional_params& options) const override;
};
}  // namespace kernel_selector
