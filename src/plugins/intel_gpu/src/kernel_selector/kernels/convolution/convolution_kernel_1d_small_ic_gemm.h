// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <algorithm>
#include <string>
#include <vector>

namespace kernel_selector {

// Implicit-GEMM convolution for a 1D convolution with a single input feature and a
// large filter, e.g. IC=1, filter length 2048, OC=1025. IC and the filter are
// merged into one reduction axis - M = OC, N = batch * output length,
// K = IC * filter length - so the feature dimension is not padded up to a blocked
// block size. The im2col matrix is never materialized.
class ConvolutionKernel_1d_small_ic_gemm : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_1d_small_ic_gemm() : Parent("convolution_gpu_1d_small_ic_gemm") {}
    ~ConvolutionKernel_1d_small_ic_gemm() override = default;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

    // The formulation is correct for any IC; 1 is the only value measured. Widening
    // needs a benchmark against ConvolutionKernel_bfyx_to_bfyx_f16.
    static constexpr size_t max_input_features = 1;
    // K = IC * filter length has to be large enough that loading a K tile pays off.
    static constexpr size_t min_filter_len = 256;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override;
    // The tile loads handle boundaries, so the input needs no pre-padded buffer.
    bool NeedPaddedInput() const override { return false; }
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;

    // The plugin may swap X and Y (see convolution_impl::get_kernel_params), so pick
    // the axis by filter extent rather than assuming one.
    static bool long_axis_is_x(const convolution_params& params) {
        return params.filterSize.x >= params.filterSize.y;
    }
    static size_t get_filter_len(const convolution_params& params) {
        return long_axis_is_x(params) ? params.filterSize.x : params.filterSize.y;
    }
};

}  // namespace kernel_selector
