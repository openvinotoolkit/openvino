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
        std::string data_size;
        switch (params.ov_input_rank) {
            case 1 :
                data_size = dims.b();
                break;
            case 2 :
                data_size = dims.f();
                break;
            case 3 :
                data_size = dims.y();
                break;
            default:
                data_size = dims.x();
                break;
        }

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
        MakeJitConstant("VEC_SIZE", vec_size),
        MakeJitConstant("VLOAD", "CAT(vload, VEC_SIZE)"),
        MakeJitConstant("VSTORE", "CAT(vstore, VEC_SIZE)"),
        MakeJitConstant("INPUT_VEC_TYPE", "MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)"),
        MakeJitConstant("ACCUMULATOR_VEC_TYPE", "MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE)"),
        MakeJitConstant("OUTPUT_VEC_TYPE", "MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)"),
        MakeJitConstant("AS_INPUT_VEC_TYPE", "CAT(as_, INPUT_VEC_TYPE)"),
        MakeJitConstant("AS_ACCUMULATOR_VEC_TYPE", "CAT(as_, ACCUMULATOR_VEC_TYPE)"),
        MakeJitConstant("TO_ACCUMULATOR_VEC_TYPE", "CAT(convert_, ACCUMULATOR_VEC_TYPE)"),
        MakeJitConstant("TO_OUTPUT_VEC_TYPE", "CAT(convert_, OUTPUT_VEC_TYPE)"),
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
        // data size to be processed within a LWG
        switch (params.ov_input_rank) {
            case 1:
                dispatchData.dataSize = input.Batch().v;
                dispatchData.dataCount = 1;
            case 2:
                dispatchData.dataSize = input.Feature().v;
                dispatchData.dataCount = input.Batch().v;
            case 3:
                dispatchData.dataSize = input.Y().v;
                dispatchData.dataCount = input.Batch().v * input.Feature().v;
                break;
            default:
                dispatchData.dataSize = input.X().v;
                dispatchData.dataCount = input.Batch().v * input.Feature().v * input.Z().v * input.Y().v;
                break;
        }

        dispatchData.slmSize = dispatchData.dataSize / vec_size;
        dispatchData.leftovers = dispatchData.dataSize % vec_size;

        dispatchData.gws[0] = dispatchData.slmSize;
        dispatchData.gws[1] = dispatchData.dataCount;
        dispatchData.gws[2] = 1;

        dispatchData.lws[0] = dispatchData.slmSize;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }
    return dispatchData;
}

bool RMSKernelBfyxOpt::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        return false;

    const rms_params& params = static_cast<const rms_params&>(p);
    const auto& gamma = params.inputs[1];

    if (!gamma.is_dynamic()) {
        size_t data_size = gamma.LogicalSize();
        if (data_size < vec_size) {
            return false;
        }
        auto local_mem_per_wi = 2 * BytesPerElement(params.inputs[0].GetDType());
        auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);
        auto slm_size = data_size / vec_size;
        if (slm_size > max_lws) {
            return false;
        }
    }

    return true;
}

KernelsData RMSKernelBfyxOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority RMSKernelBfyxOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
