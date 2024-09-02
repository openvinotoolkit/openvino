// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

bool SelectKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SELECT) {
        return false;
    }

    const select_params& params = static_cast<const select_params&>(p);

    if (params.inputs[1].GetDType() != params.inputs[2].GetDType()) {
        return false;
    }

    if (params.inputs.size() != 3) {
        return false;
    }

    return true;
}

JitConstants SelectKernelBase::GetJitConstantsCommon(const select_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    std::string inputs_decls;

    for (size_t i = 0; i < params.inputs.size(); i++) {
        std::string const_str = "const";

        inputs_decls +=
            const_str + " __global " + toCLType(params.inputs[i].GetDType()) + "* input" + toCodeString(i) + ", ";
    }

    jit.AddConstant(MakeJitConstant("INPUTS_DECLS", inputs_decls));

    std::string destType, absType;

    // i8, i8, i8
    // i8, i8, u8
    // u8, u8, i8
    // u8, u8, u8
    if ((params.inputs[0].GetDType() == Datatype::INT8 || params.inputs[0].GetDType() == Datatype::UINT8) &&
        (params.inputs[1].GetDType() == Datatype::INT8 || params.inputs[1].GetDType() == Datatype::UINT8)) {
        jit.AddConstant(MakeJitConstant("MASK", "INPUT_0"));
    } else {
        // x, x, f32
        // x, x, f16
        if (params.inputs[0].GetDType() == Datatype::F32 || params.inputs[0].GetDType() == Datatype::F16) {
            absType = "fabs";
        // f32, f32, i8
        // f32, f32, u8
        // f16, f16, i8
        // f16, f16, u8
        // i32, i32, i8
        // i32, i32, u8
        // i16, i16, i8
        // i16, i16, u8
        } else {
            absType = "abs";
        }

        // f32, f32, x
        // i32, i32, x
        if (params.inputs[1].GetDType() == Datatype::F32 || params.inputs[1].GetDType() == Datatype::INT32) {
            destType = "int";
        // f16, f16, x
        // i16, i16, x
        } else if (params.inputs[1].GetDType() == Datatype::F16 || params.inputs[1].GetDType() == Datatype::INT16) {
            destType = "short";
        // i8, i8, f32
        // i8, i8, f16
        // u8, u8, f32
        // u8, u8, f16
        } else {
            destType = "char";
        }

        jit.AddConstant(MakeJitConstant("MASK", "convert_" + destType + "_rtp(" + absType + "(INPUT_0))"));
    }

    return jit;
}

JitConstants SelectKernelBase::GetJitConstants(const select_params& params) const {
    return GetJitConstantsCommon(params);
}

SelectKernelBase::DispatchData SelectKernelBase::SetDefault(const select_params& params) const {
    DispatchData dispatchData;
    const auto& out = params.outputs[0];
    const auto& in = params.inputs[0];
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    switch (out.Dimentions()) {
    case 4:
        dispatchData.gws = { out.X().v, out.Y().v, out.Feature().v * out.Batch().v };

        dims_by_gws = {{ Tensor::DataChannelName::X },
                       { Tensor::DataChannelName::Y },
                       { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};
        break;
    case 5:
        dispatchData.gws = { out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v };

        dims_by_gws = {{ Tensor::DataChannelName::X },
                       { Tensor::DataChannelName::Y, Tensor::DataChannelName::Z },
                       { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};
        break;
    default:
        throw std::invalid_argument("Unsupported data layout for select primitive");
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in.GetLayout(), out.GetLayout(), dims_by_gws);

    return dispatchData;
}

void SelectKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const select_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData SelectKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<select_params>(params);
    select_params& newParams = *static_cast<select_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    DispatchData dispatchData = SetDefault(newParams);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false,
                     (uint32_t)newParams.inputs.size(),
                     0,
                     1,
                     newParams.is_shape_agnostic);

    return {kd};
}
}  // namespace kernel_selector
