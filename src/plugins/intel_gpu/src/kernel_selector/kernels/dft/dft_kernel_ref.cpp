// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dft_kernel_ref.h"

#include <kernel_selector_utils.h>

#include <string>
#include <vector>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const dft_params& params) {
    CommonDispatchData dispatch_data;
    const auto in_layout = params.inputs.front().GetLayout();
    const auto& output = params.outputs.front();
    const auto out_layout = output.GetLayout();
    const auto out_rank = output.Dimentions();

    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    // We are skipping X, since it contains complex pairs and always has dimension 2
    switch (out_rank) {
    case 4:
        dispatch_data.gws = {output.Y().v, output.Feature().v, output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE},
                       {Tensor::DataChannelName::BATCH}};
        break;
    case 5:
        dispatch_data.gws = {output.Y().v, output.Z().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    case 6:
        dispatch_data.gws = {output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z, Tensor::DataChannelName::W},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    default:
        throw std::invalid_argument("Unsupported output rank for dft primitive");
    }

    dispatch_data.lws =
        GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatch_data;
}

template <class T>
void MakeJitConstForParam(JitConstants& jit, const std::string& name, size_t rank, int64_t index, T value) {
    switch (index) {
    case 0:
        jit.AddConstant(MakeJitConstant(name + "_BATCH", value));
        break;
    case 1:
        jit.AddConstant(MakeJitConstant(name + "_FEATURE", value));
        break;
    case 2:
        if (rank == 6) {
            jit.AddConstant(MakeJitConstant(name + "_W", value));
        } else if (rank == 5) {
            jit.AddConstant(MakeJitConstant(name + "_Z", value));
        } else {  // rank == 4
            jit.AddConstant(MakeJitConstant(name + "_Y", value));
        }
        break;
    case 3:
        if (rank == 6) {
            jit.AddConstant(MakeJitConstant(name + "_Z", value));
        } else {  // rank == 5
            jit.AddConstant(MakeJitConstant(name + "_Y", value));
        }
        break;
    case 4:
        jit.AddConstant(MakeJitConstant(name + "_Y", value));
        break;
    default:
        throw std::invalid_argument("Unsupported index for dft primitive");
    }
}

}  // namespace

KernelsData DFTKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<dft_params>(params);
    const auto& derived_params = dynamic_cast<const dft_params&>(params);

    // For IRDFT case we create two kernels with different data
    // First, do IDFT on outer axes and input data
    // Second, do IRDFT on the last axis and data from the first kernel
    if (derived_params.mode == dft_params::Mode::real && derived_params.direction == dft_params::Direction::inverse &&
        derived_params.axes.size() > 1) {
        // Helper vector
        std::vector<std::pair<dft_params, cldnn::arguments_desc>> kernels_params;

        // Fill IDFT kernel data
        auto idft_params = derived_params;
        idft_params.mode = dft_params::Mode::complex;
        idft_params.axes.pop_back();
        idft_params.signal_size.pop_back();
        const cldnn::arguments_desc idft_arguments{{ArgumentDescriptor::Types::INPUT, 0},
                                                   {ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}};

        auto& idft_input = idft_params.inputs.front();
        auto idft_input_sizes = idft_input.LogicalDims();
        // NOTE: This is a small workaround for a 3d case
        // We always should have first dimension equal to 2, so we swap it with the second dimension
        if (idft_input_sizes[0] == 1) {
            std::swap(idft_input_sizes[0], idft_input_sizes[1]);
            idft_input = DataTensor(idft_input_sizes, idft_input.GetDType(), idft_input.GetLayout());
        }

        // Calculate IDFT output sizes
        auto idft_output_sizes = idft_input_sizes;
        auto& idft_output = idft_params.outputs.front();
        for (const auto& axis : idft_params.axes) {
            auto inverted_axis = idft_output_sizes.size() - 1 - axis;
            idft_output_sizes[inverted_axis] = idft_output.LogicalDims()[inverted_axis];
        }
        idft_output = DataTensor(idft_output_sizes, idft_input.GetDType(), idft_input.GetLayout());

        // Set internal buffer
        kd.internalBufferDataType = idft_input.GetDType();
        kd.internalBuffers.push_back(idft_output.PhysicalSizeInBytes());

        // Fill IRDFT kernel data
        auto irdft_params = derived_params;
        irdft_params.inputs.front() = idft_output;
        irdft_params.axes = {derived_params.axes.back()};
        irdft_params.signal_size = {derived_params.signal_size.back()};
        const cldnn::arguments_desc irdft_arguments{{ArgumentDescriptor::Types::INTERNAL_BUFFER, 0},
                                                    {ArgumentDescriptor::Types::OUTPUT, 0}};

        // Fill kernels
        kernels_params.emplace_back(idft_params, idft_arguments);
        kernels_params.emplace_back(irdft_params, irdft_arguments);
        const auto kKernelsNum = kernels_params.size();
        kd.kernels.resize(kKernelsNum);
        for (size_t i = 0; i < kKernelsNum; ++i) {
            dft_params kernel_params;
            cldnn::arguments_desc kernel_arguments;
            std::tie(kernel_params, kernel_arguments) = kernels_params[i];

            const auto dispatch_data = SetDefault(kernel_params);
            const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params, i);
            const auto jit_constants = GetJitConstants(kernel_params);
            const auto jit = CreateJit(kernelName, jit_constants, entry_point);
            auto& clKernelData = kd.kernels[i];
            FillCLKernelData(clKernelData, dispatch_data, kernel_params.engineInfo, kernelName, jit, entry_point);
            clKernelData.params.arguments = kernel_arguments;
        }
    } else {
        const auto dispatch_data = SetDefault(derived_params);
        const auto entry_point = GetEntryPoint(kernelName, derived_params.layerID, derived_params);
        const auto jit_constants = GetJitConstants(derived_params);
        const auto jit = CreateJit(kernelName, jit_constants, entry_point);
        auto& clKernelData = kd.kernels[0];
        FillCLKernelData(clKernelData, dispatch_data, derived_params.engineInfo, kernelName, jit, entry_point);
    }

    return {kd};
}

ParamsKey DFTKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableBatching();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

bool DFTKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::DFT) {
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
    const auto out_rank = params.outputs.front().Dimentions();
    const auto out_sizes = params.outputs.front().LogicalDims();
    const auto in_rank = params.inputs.front().Dimentions();
    const auto in_sizes = params.inputs.front().LogicalDims();
    const auto dims_size = in_sizes.size() - 1;
    auto signal_sizes = out_sizes;

    size_t s = 1;
    for (size_t i = 0; i < params.axes.size(); ++i) {
        // opencl kernels have inverted order of dimensions with respect to axis spec: x is smallest index, b is largest
        auto axis = params.axes[i];

        // when axis is negative value, convert to positive.
        if (axis < 0) {
            // RDFT has converted by r + a, others r -1 + a by op specification
            if (params.mode == dft_params::Mode::real && params.direction == dft_params::Direction::forward)
                axis = out_rank -1 + axis; // (out_rank-1) is in_rank
            else
                axis = in_rank -1 + axis;
        }

        auto inverted_axis = dims_size - axis;
        auto signal_size = params.signal_size[i];

        // For RDFT case, we need to take signal size into account, as output size can be not the same as signal size
        if (params.mode == dft_params::Mode::real && params.direction == dft_params::Direction::forward) {
            if (signal_size != -1) {
                signal_sizes[inverted_axis] = signal_size;
            } else {
                signal_sizes[inverted_axis] = in_sizes[inverted_axis];
            }
        }

        s *= signal_sizes[inverted_axis];

        // NOTE: We can use full signal size as axis value, but this doesn't make much sense, as it will be zero-padded
        // So, we take minimum size here and save some dummy cycles in kernel
        auto axis_value = std::min(signal_sizes[inverted_axis], in_sizes[inverted_axis]);

        // For IRDFT case, we should use full signal size as axis value and interpret input data as Hermitian-symmetric
        if (params.mode == dft_params::Mode::real && params.direction == dft_params::Direction::inverse) {
            axis_value = signal_sizes[inverted_axis];
            MakeJitConstForParam(jit, "SYMMETRIC_AXIS", out_rank, axis, true);
        }

        MakeJitConstForParam(jit, "AXIS", out_rank, axis, axis_value);
        MakeJitConstForParam(jit, "SIGNAL_SIZE", out_rank, axis, signal_sizes[inverted_axis]);
    }
    if (params.direction == dft_params::Direction::inverse) {
        jit.AddConstant(MakeJitConstant("INVERSE_DFT_MULTIPLIER", 1.f / s));
    }
    if (params.mode == dft_params::Mode::real) {
        jit.AddConstant(MakeJitConstant("REAL_DFT", true));
    }
    return jit;
}

}  // namespace kernel_selector
