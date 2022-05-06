// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dft_kernel_ref.h"

#include <kernel_selector_utils.h>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const dft_params& params) {
    CommonDispatchData dispatchData;

    auto outDims = params.outputs.front().LogicalDims();
    // opencl kernels have inverted order of dimensions with respect to axis spec: x is smallest index, b is largest
    // we are skipping x, since it contains complex pairs
    auto complexSize = std::accumulate(outDims.begin() + 1, outDims.end(), size_t{1}, std::multiplies<size_t>{});
    dispatchData.gws = {1, 1, complexSize};  // TODO: these could be split better
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

}  // namespace

KernelsData DFTKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelsData kernels_data;
    if (!Validate(params, options)) {
        return kernels_data;
    }
    kernels_data.push_back(KernelData::Default<dft_params>(params));
    KernelData& kernel_data = kernels_data.front();
    auto& derived_params = dynamic_cast<dft_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(derived_params);
    auto entry_point = GetEntryPoint(kernelName, derived_params.layerID, params, options);
    auto jit_constants = GetJitConstants(derived_params);
    auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& clKernelData = kernel_data.kernels[0];
    FillCLKernelData(clKernelData, dispatch_data, params.engineInfo, kernelName, jit, entry_point);
    return kernels_data;
}

KernelsPriority DFTKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey DFTKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableBatching();
    return k;
}

bool DFTKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::DFT || o.GetType() != KernelType::DFT) {
        return false;
    }

    auto& params = dynamic_cast<const dft_params&>(p);
    if (params.inputs.size() != 1) {
        return false;
    }

    return true;
}

JitConstants DFTKernelRef::GetJitConstants(const dft_params& params) const {
    auto jit_constants = MakeBaseParamsJitConstants(params);
    const auto output_sizes = params.outputs.front().LogicalDims();
    const auto input_sizes = params.inputs.front().LogicalDims();
    const auto n1 = input_sizes.size() - 1;
    for (auto axis : params.axes) {
        axis = n1 - axis;  // opencl kernels have inverted order of dimensions with respect to axis spec: x is smallest
                           // index, b is largest
        jit_constants.AddConstant(
            MakeJitConstant('A' + std::to_string(axis), std::min(output_sizes[axis], input_sizes[axis])));
    }
    if (params.kind == dft_params::inverse) {
        size_t s = 1;
        for (auto axis : params.axes) {
            axis = n1 - axis;  // opencl kernels have inverted order of dimensions with respect to axis spec: x is
                               // smallest index, b is largest
            s *= output_sizes[axis];
        }
        jit_constants.AddConstant(MakeJitConstant("INVERSE_DFT_MULTIPLIER", 1.f / s));
    }
    return jit_constants;
}

}  // namespace kernel_selector
