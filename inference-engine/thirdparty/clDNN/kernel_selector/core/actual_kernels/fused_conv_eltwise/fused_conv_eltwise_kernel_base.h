/*
// Copyright (c) 2016-2020 Intel Corporation
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

#include "weight_bias_kernel_base.h"
#include "actual_kernels/convolution/convolution_params.h"
#include "actual_kernels/eltwise/eltwise_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fused_conv_eltwise_params : public weight_bias_params {
    fused_conv_eltwise_params() : weight_bias_params(KernelType::FUSED_CONV_ELTWISE) {}

    struct conv_data {
        uSize filterSize;
        uSize stride;
        uSize dilation;
        uSize padding;
        uint32_t split = 1;
        bool depthwise_separable_opt = false;
        bool transposed = false;
        bool int8_quantization = false;
        bool output_calibration = false;
        bool local_convolution = false;
        float input_quantization_factor = 1.0f;
        float output_quantization_factor = 1.0f;
        MultiDataTensor weights_quantization_factors;
        MultiDataTensor output_calibration_factors;

        std::vector<base_activation_params> activations;
    } conv;

    struct eltw_data {
        std::vector<eltwise_params::Node> operations;
        std::vector<float> coefficients;
        std::vector<eltwise_params::UpdateInputData> updateInputIds;
        std::vector<uSize> stride;

        bool layoutBased = false;
        bool int8_quantization = false;
        bool output_calibration = false;
        float output_quantization_factor = 1.0f;

        MultiDataTensor output_calibration_factors;
    } eltw;

    float non_conv_scale = 1.0f;
    bool second_input_in_output = false;
    bool depth_to_space_already_fused = false;

    std::string to_string() const override;
    std::string to_cache_string_v2() const override;
    ParamsKey GetParamsKey() const override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fused_conv_eltwise_optional_params : weight_bias_optional_params {
    fused_conv_eltwise_optional_params() : weight_bias_optional_params(KernelType::FUSED_CONV_ELTWISE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ConvolutionKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class fused_conv_eltwise_kernel_base : public WeightBiasKernelBase {
public:
    using WeightBiasKernelBase::WeightBiasKernelBase;
    virtual ~fused_conv_eltwise_kernel_base() {}

    struct DispatchData : public CommonDispatchData {
        struct CLDNNStyle {
            size_t blockWidth;
            size_t blockHeight;  // used for kernels processing blocks
            size_t prefetch;
            size_t inputBlockArraySize;  // Number of elements in array of UNIT_TYPE that must be specified in kernel to
                                         // store/cache input block.
            size_t inputBlockWidth;      // Number of elements in X dimension stored/cached in input block.
        };

        struct GEMMStyle {
            size_t subBlockDimM;
            size_t subBlockDimK;
            size_t subBlockDimN;
            size_t globalWorkSizeDX;
            size_t globalWorkSizeDY;
            size_t globalWorkSizeDZ;
        };

        union {
            CLDNNStyle cldnnStyle;
            GEMMStyle gemmStyle;
        };
    };

    std::string GetAutoTuneOptions(int autoTuneIndex) const;
    std::vector<std::string> autoTuneOptions = {DEFAULT, NO_PRERA_SCH, AGE_BASED};
    KernelsData GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                                   const optional_params& options,
                                                   int autoTuneIndex = -1) const override;

protected:
    virtual WeightsLayout GetPreferreddWeightsLayout(const fused_conv_eltwise_params &) const = 0;
    virtual std::string GetKernelName(const fused_conv_eltwise_params&) const { return kernelName; }
    virtual bool NeedPaddedInput() const { return false; }
    bool Validate(const Params& p, const optional_params& o) const override;
    virtual JitConstants GetJitConstants(const fused_conv_eltwise_params& params, const DispatchData& kd) const;
    virtual DispatchData SetDefault(const fused_conv_eltwise_params& params, int autoTuneIndex = -1) const;
    static bool CheckWorkGroups(const DispatchData&);
    static bool CheckPitchForSplitOnly(const fused_conv_eltwise_params& params);
    KernelsData GetCommonKernelsData(const Params& params,
                                     const optional_params& options,
                                     const std::string exeMode = DEFAULT,
                                     int autoTuneIndex = -1) const;
};

bool FusedConvolutionEltwiseCheckInput(const Params& p, const optional_params& o);
bool CheckConvolutionPaddedInputDesc(const fused_conv_eltwise_params& params, const DataTensor& reqDesc);
bool CovolutionUpdateInputParams(fused_conv_eltwise_params& params);

}  // namespace kernel_selector
