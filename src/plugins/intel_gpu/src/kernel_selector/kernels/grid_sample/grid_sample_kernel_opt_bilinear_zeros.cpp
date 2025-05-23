// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel_opt_bilinear_zeros.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

constexpr size_t THREADS_PER_BLOCK = 256;
constexpr size_t GRID_ITEMS_PER_BLOCK = THREADS_PER_BLOCK;

CommonDispatchData GridSampleKernelOpt_BilinearZeros::CalcDispatch(const grid_sample_params& kernel_params) const {
    CommonDispatchData dispatch_data;
    const auto& output = kernel_params.outputs.front();

    auto blocks = (output.Y().v * output.X().v + GRID_ITEMS_PER_BLOCK - 1) / GRID_ITEMS_PER_BLOCK;

    dispatch_data.gws = {output.Batch().v, blocks * THREADS_PER_BLOCK, 1};
    dispatch_data.lws = {1, THREADS_PER_BLOCK, 1};

    return dispatch_data;
}

KernelsPriority GridSampleKernelOpt_BilinearZeros::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}

bool GridSampleKernelOpt_BilinearZeros::Validate(const Params& params) const {
    if (!TBase::Validate(params))
        return false;

    const auto& kernel_params = static_cast<const grid_sample_params&>(params);
    if (kernel_params.interpolation_mode != grid_sample_params::InterpolationMode::BILINEAR)
        return false;

    if (kernel_params.padding_mode != grid_sample_params::PaddingMode::ZEROS)
        return false;

    return true;
}

JitConstants GridSampleKernelOpt_BilinearZeros::GetJitConstants(const grid_sample_params& kernel_params) const {
    auto jit_constants = TBase::GetJitConstants(kernel_params);

    jit_constants.AddConstants({
        MakeJitConstant("GRID_ITEMS_PER_BLOCK", GRID_ITEMS_PER_BLOCK)
    });

    return jit_constants;
}

ParamsKey GridSampleKernelOpt_BilinearZeros::GetSupportedKey() const {
    ParamsKey key;
    key.EnableAllInputDataType();
    key.EnableAllOutputDataType();
    key.EnableDifferentTypes();
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    return key;
}

}  // namespace kernel_selector
