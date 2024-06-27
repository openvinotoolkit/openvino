// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nonzero_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey GatherNonzeroKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::UINT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::UINT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants GatherNonzeroKernelRef::GetJitConstants(const gather_nonzero_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    const auto& input = params.inputs[0];
    jit.AddConstant(MakeJitConstant("OV_INPUT_RANK", params.ov_input_rank));

    auto max_local_mem_size = params.engineInfo.maxLocalMemSize / (params.outputs[0].ElementSize());
    jit.AddConstant(MakeJitConstant("MAX_LOCAL_MEM_SIZE", max_local_mem_size));

    if (input.is_dynamic()) {
        DimensionAccessHelperJit dims(input);
        const std::string total_data_size = toVectorMulString({dims.x(), dims.y(), dims.z(), dims.w(), dims.f(), dims.b()});
        jit.AddConstant(MakeJitConstant("TOTAL_DATA_SIZE", total_data_size));
    } else {
        jit.AddConstant(MakeJitConstant("TOTAL_DATA_SIZE", params.inputs[0].LogicalSize()));
        if (params.inputs[0].LogicalSize() * params.ov_input_rank < max_local_mem_size) {
            jit.AddConstant(MakeJitConstant("USE_LOCAL_MEM", 1));
        }
    }
    return jit;
}

CommonDispatchData GatherNonzeroKernelRef::SetDefault(const gather_nonzero_params& params) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = {1, 1, 1};
    dispatchData.lws = {1, 1, 1};

    return dispatchData;
}

void GatherNonzeroKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const gather_nonzero_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData GatherNonzeroKernelRef::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::GATHER_NONZERO);

    KernelData kd = KernelData::Default<gather_nonzero_params>(params);
    gather_nonzero_params& newParams = *static_cast<gather_nonzero_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    GetUpdateDispatchDataFunc(kd);

    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     2,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     newParams.is_shape_agnostic);

    return {kd};
}

KernelsPriority GatherNonzeroKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool GatherNonzeroKernelRef::Validate(const Params& p) const {
    if (!KernelBaseOpenCL::Validate(p))
        return false;

    const auto& rp = static_cast<const gather_nonzero_params&>(p);

    return Tensor::SimpleLayout(rp.inputs[0].GetLayout());
}
}  // namespace kernel_selector
