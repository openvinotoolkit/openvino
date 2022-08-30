// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "weight_bias_kernel_base.h"
#include "binary_convolution_params.h"
#include <string>
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BinaryConvolutionKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BinaryConvolutionKernelBase : public WeightBiasKernelBase {
public:
    using WeightBiasKernelBase::WeightBiasKernelBase;
    virtual ~BinaryConvolutionKernelBase() {}

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
    virtual WeightsLayout GetPreferredWeightLayout(const binary_convolution_params &) const = 0;
    virtual std::string GetKernelName(const binary_convolution_params&) const { return kernelName; }
    virtual bool NeedPaddedInput() const { return false; }
    bool Validate(const Params& p, const optional_params& o) const override;
    virtual JitConstants GetJitConstants(const binary_convolution_params& params, const DispatchData& dispatchData) const;
    virtual JitConstants GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                        const DispatchData& dispatchData) const;
    virtual DispatchData SetDefault(const binary_convolution_params& params, int autoTuneIndex = -1) const;
    static bool CheckWorkGroups(const DispatchData&);
    KernelsData GetCommonKernelsData(const Params& params,
                                     const optional_params& options,
                                     const std::string exeMode = DEFAULT,
                                     int autoTuneIndex = -1) const;
};

bool ConvolutionBinaryCheckInput(const Params& p, const optional_params& o);
bool CheckConvolutionBinaryPaddedInputDesc(const binary_convolution_params& params, const DataTensor& reqDesc);
bool ConvolutionBinaryUpdateInputParams(binary_convolution_params& params);

}  // namespace kernel_selector
