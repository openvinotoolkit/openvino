// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deformable_convolution_kernel_bfyx_interp.h"
#include <string>

namespace kernel_selector {

ParamsKey DeformableConvolutionKernel_bfyx_interp::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
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
    k.EnableGroupedConvolution();
    k.EnableDeformableMode();
    k.EnableDeformableMask();
    k.EnableBilinearInterpolationPad();
    return k;
}

DeviceFeaturesKey DeformableConvolutionKernel_bfyx_interp::get_required_device_features_key(const Params& params, const optional_params& options) const {
    DeviceFeaturesKey k;
    k.requires_reqd_subgroup_size();

    return k;
}

CommonDispatchData DeformableConvolutionKernel_bfyx_interp::SetDefault(const convolution_params& params) const {
    CommonDispatchData dispatchData;

    const auto& out = params.outputs[0];

    auto x = out.X().v;
    auto y = out.Y().v;
    auto b = out.Batch().v;
    auto kernel_size = params.kernelSize.x * params.kernelSize.y;

    dispatchData.gws[0] = Align(x * y, 16);
    dispatchData.gws[1] = params.deformable_groups * b;
    dispatchData.gws[2] = kernel_size;

    dispatchData.lws[0] = 16;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority DeformableConvolutionKernel_bfyx_interp::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_2;
}

JitConstants DeformableConvolutionKernel_bfyx_interp::GetJitConstants(const convolution_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", 16));
    jit.AddConstant(MakeJitConstant("FILTER_SIZE_X", params.kernelSize.x));
    jit.AddConstant(MakeJitConstant("FILTER_SIZE_Y", params.kernelSize.y));
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

KernelsData DeformableConvolutionKernel_bfyx_interp::GetKernelsData(const Params& params,
                                                                    const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    CommonDispatchData dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, EXE_MODE_DEFAULT,
                     false, false, static_cast<int>(newParams.inputs.size()));

    return {kd};
}
}  // namespace kernel_selector
