// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey RMSKernelBfyxOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants RMSKernelBfyxOpt::GetJitConstants(const rms_params& params, DispatchData dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    if (params.has_dynamic_tensors()) {
        const auto& input = params.inputs[0];
        DimensionAccessHelper dims(input);
        const std::string data_size = toVectorMulString({dims.x(), dims.y(), dims.z()});
        const std::string lws_0 = "get_local_size(0)";
        jit.AddConstants({
            MakeJitConstant("DATA_SIZE", data_size),
            MakeJitConstant("LWS", lws_0),
            MakeJitConstant("SLM_SIZE", dispatchData.maxSlmSize)
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("DATA_SIZE", dispatchData.dataSize),
            MakeJitConstant("LWS", dispatchData.slmSize),
            MakeJitConstant("SLM_SIZE", dispatchData.slmSize),
            MakeJitConstant("LEFTOVERS", dispatchData.leftovers)
        });
    }
    jit.AddConstants({
        MakeJitConstant("VEC_SIZE", 8),
        MakeJitConstant("VLOAD", "CAT(vload, VEC_SIZE)"),
        MakeJitConstant("VSTORE", "CAT(vstore, VEC_SIZE)"),
        MakeJitConstant("INPUTVTYPE", "CAT(INPUT0_TYPE, VEC_SIZE)"),
        MakeJitConstant("OUTPUTVTYPE", "CAT(OUTPUT_TYPE, VEC_SIZE)"),
        MakeJitConstant("AS_INPUTVTYPE", "CAT(as_, INPUTVTYPE)"),
        MakeJitConstant("AS_OUTPUTVTYPE", "CAT(as_, OUTPUTVTYPE)")
    });

    return jit;
}

RMSKernelBase::DispatchData RMSKernelBfyxOpt::SetDefault(const rms_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];

    auto local_mem_per_wi = 2 * BytesPerElement(params.inputs[0].GetDType());
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);
    dispatchData.maxSlmSize = max_lws;

    if (!params.has_dynamic_tensors()) {
        dispatchData.dataSize = input.X().v * input.Y().v * input.Z().v;
        dispatchData.dataCount = input.Batch().v * input.Feature().v;
        dispatchData.slmSize = dispatchData.dataSize / 8;
        dispatchData.leftovers = dispatchData.dataSize % 8;

        dispatchData.gws[0] = dispatchData.slmSize;
        dispatchData.gws[1] = dispatchData.dataCount;
        dispatchData.gws[2] = 1;

        dispatchData.lws[0] = dispatchData.slmSize;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }
    return dispatchData;
}

bool RMSKernelBfyxOpt::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o))
        return false;

    const rms_params& params = static_cast<const rms_params&>(p);
    const auto& gamma = params.inputs[1];

    if (!gamma.is_dynamic() && (gamma.LogicalSize() < 8))
        return false;

    return true;
}

KernelsData RMSKernelBfyxOpt::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority RMSKernelBfyxOpt::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
