// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <algorithm>
#include <string>
#include <vector>

namespace kernel_selector {

// Implicit-GEMM convolution for 1D convolutions with a single input feature and
// a large tap count, e.g. IC=1, taps=2048, OC=1025.
//
// The blocked-layout kernels pad the feature dimension up to the block size (16),
// so an IC=1 convolution wastes 15/16 of every MAC it issues. This kernel merges
// IC and the taps into one reduction axis instead, giving
//
//   M = OC, N = batch * output_length, K = IC * taps
//
// so there is no feature padding and K is large enough to amortize the tile
// loads. The im2col matrix is never materialized: K maps back to
// (input feature, tap) and N to (batch, output position) on the fly while
// loading tiles into SLM.
class ConvolutionKernel_1d_small_ic_gemm : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_1d_small_ic_gemm() : Parent("convolution_gpu_1d_small_ic_gemm") {}
    ~ConvolutionKernel_1d_small_ic_gemm() override = default;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

    // 1 because that is the only value measured, not a limit of the formulation:
    // the kernel decomposes K into (input feature, tap) and is correct for any IC.
    // At IC > 1 the competitor is ConvolutionKernel_bfyx_to_bfyx_f16, whose input
    // is planar and pays no padding cost, so widening needs a benchmark against it
    // plus kernel-side accuracy coverage. Until then
    // convolution_1d_small_ic_gemm_f32.rejects_input_features_above_max repeats
    // this bound as a literal so widening cannot pass silently.
    static constexpr size_t max_input_features = 1;
    // K = IC * taps has to be large enough that loading a K tile pays for itself.
    static constexpr size_t min_taps = 256;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override;
    // Boundary conditions are handled by the tile loads, so the input does not
    // have to be pre-padded into a larger buffer.
    bool NeedPaddedInput() const override { return false; }
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;

    // The plugin swaps X and Y so a 1D convolution's long axis lands on X (see
    // convolution_impl::get_kernel_params), consistently across the tensors,
    // filterSize, padding, stride and dilation. Picking whichever axis is
    // non-degenerate makes the kernel independent of whether the swap happened;
    // Validate() rejects anything where the two disagree.
    static bool long_axis_is_x(const convolution_params& params) {
        return params.filterSize.x >= params.filterSize.y;
    }
    static size_t get_taps(const convolution_params& params) {
        return long_axis_is_x(params) ? params.filterSize.x : params.filterSize.y;
    }
};

}  // namespace kernel_selector
