// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deformable_convolution_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
namespace {

DeformableConvolutionKernel_bfyx_opt::DispatchData set_default(const convolution_params& params, int idx) {
    DeformableConvolutionKernel_bfyx_opt::DispatchData dispatchData;

    const auto& out = params.outputs[0];
    if (idx == 0) {
        auto x = out.X().v;
        auto y = out.Y().v;
        auto b = out.Batch().v;
        auto kernel_size = params.filterSize.x * params.filterSize.y;

        dispatchData.gws[0] = Align(x * y, 16);
        dispatchData.gws[1] = params.deformable_groups * b;
        dispatchData.gws[2] = kernel_size;

        dispatchData.lws[0] = 16;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    } else if (idx == 1) {
        auto x = out.X().v;
        auto y = out.Y().v;
        auto f = out.Feature().v;
        auto b = out.Batch().v;

        dispatchData.gws[0] = CeilDiv(x * y, 16);
        dispatchData.gws[1] = Align(f, 16);
        dispatchData.gws[2] = b;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 16;
        dispatchData.lws[2] = 1;
    }

    return dispatchData;
}
}  // namespace

ParamsKey DeformableConvolutionKernel_bfyx_opt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDeformableMode();
    k.EnableDeformableMask();
    k.EnableBilinearInterpolationPad();
    return k;
}

DeviceFeaturesKey DeformableConvolutionKernel_bfyx_opt::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

KernelsPriority DeformableConvolutionKernel_bfyx_opt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

JitConstants DeformableConvolutionKernel_bfyx_opt::GetJitConstants(const convolution_params& params,
                                                                    const DispatchData& /*dispatchData*/) const {
    JitConstants jit = WeightBiasKernelBase::GetJitConstants(params);
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", 16));
    jit.AddConstant(MakeJitConstant("INPUT_CHANNELS", params.inputs[0].Feature().v));
    jit.AddConstant(MakeJitConstant("INTERPOLATED", params.intermediate_tensor));
    jit.AddConstants({MakeJitConstant("STRIDE", params.stride),
                      MakeJitConstant("PADDING", params.padding_begin),
                      MakeJitConstant("DILATION", params.dilation)
                     });
    jit.AddConstants({MakeJitConstant("DEFORMABLE_GROUPS", params.deformable_groups)});
    jit.AddConstants({MakeJitConstant("DEFORMABLE_MODE", params.deformable_mode)});
    jit.AddConstants({MakeJitConstant("DEFORMABLE_MASK_ENABLED", params.deformable_mask_enabled)});
    jit.AddConstants({MakeJitConstant("BILINEAR_INTERPOLATION_PAD", params.bilinear_interpolation_pad)});

    return jit;
}

void DeformableConvolutionKernel_bfyx_opt::SetKernelArguments(const convolution_params& params, clKernelData& kernel, size_t idx) const {
    switch (idx) {
    case 0:
        for (size_t i = 0; i < params.inputs.size(); i++) {
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(i) });
        }
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 }); // save out to temporary internal buffer
        break;

    case 1:
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 }); // out of prev stage
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 }); // real output
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::WEIGHTS, 0 });
        if (!params.bias.empty()) {
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
        }

        break;
    default:
        throw std::invalid_argument("Deformable conv has 2 kernels. valid index is 0 ~ 1.");
    }
}

KernelsData DeformableConvolutionKernel_bfyx_opt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    constexpr size_t kKernelsNum = 2;
    KernelData kd = KernelData::Default<convolution_params>(params, kKernelsNum);
    const auto& conv_params = static_cast<const convolution_params&>(params);
    if (!conv_params.deformable_mode)
        return {};

    auto preferredWeightsLayout = GetPreferredWeightsLayout(conv_params);
    bool succeed = UpdateWeightsParams(*static_cast<convolution_params*>(kd.params.get()),
                                       preferredWeightsLayout,
                                       kd.weightsReorderParams,
                                       GetSupportedKey(),
                                       conv_params.groups,
                                       conv_params.transposed);

    bool bSupportedWeightsLayout = conv_params.weights.GetLayout() == preferredWeightsLayout;
    const bool bWeightsOK = bSupportedWeightsLayout || conv_params.allowStaticInputReordering;

    if (!succeed || !bWeightsOK) {
        return {};
    }

    kd.internalBufferSizes.push_back(conv_params.intermediate_tensor.PhysicalSizeInBytes());
    kd.internalBufferDataType = conv_params.intermediate_tensor.GetDType();

    for (size_t i = 0; i < kKernelsNum; i++) {
        DispatchData dispatchData = set_default(conv_params, static_cast<int>(i));
        auto entry_point = GetEntryPoint(kernelName, conv_params.layerID, params, i);
        auto cldnn_jit = GetJitConstants(conv_params, dispatchData);
        cldnn_jit.AddConstant(MakeJitConstant("DEFORMABLE_CONV_STAGE_" + std::to_string(i), true));

        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[i];
        KernelBase::CheckDispatchData(kernelName, dispatchData, params.engineInfo.maxWorkGroupSize);
        kernel.params.workGroups.global = dispatchData.gws;
        kernel.params.workGroups.local  = dispatchData.lws;
        kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
        SetKernelArguments(conv_params, kernel, i);
    }

    return {kd};
}
}  // namespace kernel_selector
