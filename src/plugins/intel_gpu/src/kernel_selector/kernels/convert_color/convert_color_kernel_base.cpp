// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

bool ConvertColorKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::CONVERT_COLOR) {
        return false;
    }

    const convert_color_params& params = static_cast<const convert_color_params&>(p);

    if (params.inputs[0].Dimentions() > 4)
        return false;

    return true;
}

CommonDispatchData ConvertColorKernelBase::SetDefault(const convert_color_params& params) const {
    CommonDispatchData dispatchData;
    const auto& out = params.outputs[0];
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();

    dispatchData.gws = { out.Batch().v, out.Feature().v, out.Y().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout);

    return dispatchData;
}

JitConstants ConvertColorKernelBase::GetJitConstants(const convert_color_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("INPUTS_COUNT", params.inputs.size()));

    switch (params.input_color_format) {
        case color_format::NV12:
            jit.AddConstant(MakeJitConstant("CONVERT_FROM_NV12", ""));
            break;
        case color_format::I420:
            jit.AddConstant(MakeJitConstant("CONVERT_FROM_I420", ""));
            break;
        default:
            OPENVINO_THROW("Not supported input color format");
    }

    switch (params.output_color_format) {
        case color_format::RGB:
            jit.AddConstant(MakeJitConstant("CONVERT_TO_RGB", ""));
            break;
        case color_format::BGR:
            jit.AddConstant(MakeJitConstant("CONVERT_TO_BGR", ""));
            break;
        default:
            OPENVINO_THROW("Not supported output color format");
    }

    switch (params.mem_type) {
        case memory_type::buffer:
            jit.AddConstant(MakeJitConstant("BUFFER_MEM", ""));
            break;
        case memory_type::image:
            jit.AddConstant(MakeJitConstant("SURFACE_MEM", ""));
            break;
        default:
            OPENVINO_THROW("Not supported memory type");
    }
    return jit;
}

KernelsData ConvertColorKernelBase::GetCommonKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<convert_color_params>(params);
    const auto& prim_params = static_cast<const convert_color_params&>(params);

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto cldnn_jit = GetJitConstants(prim_params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    uint32_t number_of_inputs = static_cast<uint32_t>(prim_params.inputs.size());
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, number_of_inputs);

    return { kd };
}
}  // namespace kernel_selector
