// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dft_kernel_ref.h"

#include <kernel_selector_utils.h>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const dft_params& params) {
    CommonDispatchData dispatch_data;
    const auto in_layout = params.inputs.front().GetLayout();
    const auto& output = params.outputs.front();
    const auto out_layout = output.GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    // We are skipping X, since it contains complex pairs and always has dimension 2
    switch (out_layout) {
    case DataLayout::bfyx:
        dispatch_data.gws = {output.Y().v, output.Feature().v, output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE},
                       {Tensor::DataChannelName::BATCH}};
        break;
    case DataLayout::bfzyx:
        dispatch_data.gws = {output.Y().v, output.Z().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    case DataLayout::bfwzyx:
        dispatch_data.gws = {output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z, Tensor::DataChannelName::W},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    default:
        throw std::invalid_argument("Unsupported data layout for dft primitive");
    }

    dispatch_data.lws =
        GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatch_data;
}

template <class T>
void MakeJitConstForAxis(JitConstants& jit, const DataLayout& layout, int64_t index, T value) {
    std::string name = "AXIS";
    switch (index) {
    case 0:
        jit.AddConstant(MakeJitConstant(name + "_BATCH", value));
        break;
    case 1:
        jit.AddConstant(MakeJitConstant(name + "_FEATURE", value));
        break;
    case 2:
        if (layout == DataLayout::bfwzyx) {
            jit.AddConstant(MakeJitConstant(name + "_W", value));
        } else if (layout == DataLayout::bfzyx) {
            jit.AddConstant(MakeJitConstant(name + "_Z", value));
        } else {  // DataLayout::bfyx
            jit.AddConstant(MakeJitConstant(name + "_Y", value));
        }
        break;
    case 3:
        if (layout == DataLayout::bfwzyx) {
            jit.AddConstant(MakeJitConstant(name + "_Z", value));
        } else {  // DataLayout::bfzyx
            jit.AddConstant(MakeJitConstant(name + "_Y", value));
        }
        break;
    case 4:
        jit.AddConstant(MakeJitConstant(name + "_Y", value));
        break;
    default:
        throw std::invalid_argument("Unsupported axis for dft primitive");
    }
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
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableBatching();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
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
    auto jit = MakeBaseParamsJitConstants(params);
    const auto out_layout = params.outputs.front().GetLayout();
    const auto out_sizes = params.outputs.front().LogicalDims();
    const auto in_sizes = params.inputs.front().LogicalDims();

    // We are skipping X, since it contains complex pairs and should not be in axes
    const auto dims_size = in_sizes.size() - 1;

    size_t s = 1;
    for (auto axis : params.axes) {
        // opencl kernels have inverted order of dimensions with respect to axis spec: x is smallest index, b is largest
        auto inverted_axis = dims_size - axis;
        s *= out_sizes[inverted_axis];
        MakeJitConstForAxis(jit, out_layout, axis, std::min(out_sizes[inverted_axis], in_sizes[inverted_axis]));
    }
    if (params.kind == dft_params::inverse) {
        jit.AddConstant(MakeJitConstant("INVERSE_DFT_MULTIPLIER", 1.f / s));
    }
    return jit;
}

}  // namespace kernel_selector
