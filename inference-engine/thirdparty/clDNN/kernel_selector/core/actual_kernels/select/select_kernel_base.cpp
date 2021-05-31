// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

bool SelectKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::SELECT || o.GetType() != KernelType::SELECT) {
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
            const_str + " __global " + toCLType(params.inputs[i].GetDType()) + "* input" + std::to_string(i) + ", ";
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

    const auto& out = params.output;

    std::vector<size_t> gws;
    for (const auto& o : out.GetDims()) {
        gws.push_back(o.v);
    }

    for (size_t i = gws.size(); i < 4; i++) {
        gws.push_back(1U);
    }

    dispatchData.gws[0] = gws[0];
    dispatchData.gws[1] = gws[1];
    dispatchData.gws[2] = gws[2] * gws[3];
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData SelectKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<select_params>(params);
    select_params& newParams = *static_cast<select_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    DispatchData dispatchData = SetDefault(newParams);

    auto& kernel = kd.kernels[0];

    kernel.workGroups.global = dispatchData.gws;
    kernel.workGroups.local = dispatchData.lws;

    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

    return {kd};
}
}  // namespace kernel_selector
