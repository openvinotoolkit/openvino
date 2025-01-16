// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "weight_bias_kernel_base.h"
#include "convolution_params.h"
#include <string>
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ConvolutionKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ConvolutionKernelBase : public WeightBiasKernelBase {
public:
    using WeightBiasKernelBase::WeightBiasKernelBase;
    virtual ~ConvolutionKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        struct CLDNNStyle {
            size_t blockWidth, blockHeight;  // used for kernels processing blocks
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

        struct BlockParams {
            size_t output_block_width;
            size_t output_block_height;
            size_t output_block_depth;

            size_t output_block_features;

            size_t input_block_width;
            size_t input_block_height;
            size_t input_block_depth;

            size_t feature_slm_split;
        };

        CLDNNStyle cldnnStyle;
        GEMMStyle gemmStyle;
        BlockParams blockParams;
    };

    std::string GetAutoTuneOptions(int autoTuneIndex) const;
    std::vector<std::string> autoTuneOptions = {EXE_MODE_DEFAULT, EXE_MODE_NO_PRERA_SCH, EXE_MODE_AGE_BASED};
    KernelsData GetKernelsDataForAutoTune(const Params& params) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params& params, int autoTuneIndex = -1) const override;

protected:
    virtual WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const = 0;
    virtual std::string GetKernelName(const convolution_params&) const { return kernelName; }
    virtual bool NeedPaddedInput() const { return false; }
    bool Validate(const Params& p) const override;
    using WeightBiasKernelBase::GetJitConstants;
    JitConstants GetJitConstantsWithLoopUnroll(const convolution_params& params, const DispatchData& dispatchData) const;
    virtual JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const;
    virtual JitConstants GetFusedPrimitivesJitConstants(const convolution_params& params, const DispatchData& dispatchData) const;
    virtual DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const;
    static bool CheckWorkGroups(const DispatchData&);
    KernelsData GetCommonKernelsData(const Params& params,
                                     const std::string exeMode = EXE_MODE_DEFAULT,
                                     int autoTuneIndex = -1) const;

    Datatype GetPackedType(Datatype dt, size_t pack_size = 4) const;
    Datatype GetPackedInputType(const convolution_params& params) const;
    Datatype GetPackedOutputType(const convolution_params& params) const;
    Datatype GetActivationType(const convolution_params& params) const;
    Datatype GetAccumulatorType(const convolution_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

bool ConvolutionCheckInput(const Params& p);
bool CheckConvolutionPaddedInputDesc(const convolution_params& params, const DataTensor& reqDesc);
bool CheckConvolutionExplicitPaddings(const convolution_params& conv_params);
bool ConvolutionUpdateInputParams(convolution_params& params);

}  // namespace kernel_selector
