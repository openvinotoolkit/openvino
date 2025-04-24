// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "deconvolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class DeconvolutionKernel_imad_along_f_tile_bfx : public DeconvolutionKernelBase {
public:
    using Parent = DeconvolutionKernelBase;
    DeconvolutionKernel_imad_along_f_tile_bfx() : DeconvolutionKernelBase("deconvolution_gpu_imad_along_f_tile_bfx") {}
    virtual ~DeconvolutionKernel_imad_along_f_tile_bfx() = default;

    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params& p) const override;
    WeightsLayout GetPreferredWeightsLayout(const deconvolution_params &params) const override;
    CommonDispatchData SetDefault(const deconvolution_params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    JitConstants GetJitConstants(const deconvolution_params& params) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::QUANTIZE
        };
    }

    size_t GetTileIFM(const deconvolution_params& params) const;
    size_t GetTileOFM(const deconvolution_params& params) const;
    size_t GetTileX(const deconvolution_params& params) const;
    size_t GetTileB(const deconvolution_params& params) const;
};

}  // namespace kernel_selector
