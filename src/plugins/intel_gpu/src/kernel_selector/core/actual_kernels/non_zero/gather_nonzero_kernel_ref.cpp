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
    return k;
}

KernelsData GatherNonzeroKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::GATHER_NONZERO);

    KernelData kd = KernelData::Default<gather_nonzero_params>(params);
    gather_nonzero_params& newParams = *static_cast<gather_nonzero_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = MakeBaseParamsJitConstants(newParams);
    cldnn_jit.AddConstant(MakeJitConstant("OV_INPUT_RANK", newParams.ov_input_rank));
    cldnn_jit.AddConstant(MakeJitConstant("TOTAL_DATA_SIZE", newParams.inputs[0].LogicalSize()));
    auto local_mem_size = params.engineInfo.maxLocalMemSize / (newParams.outputs[0].ElementSize());
    if (newParams.inputs[0].LogicalSize() * newParams.ov_input_rank < local_mem_size) {
        cldnn_jit.AddConstant(MakeJitConstant("MAX_LOCAL_MEM_SIZE", local_mem_size));
        cldnn_jit.AddConstant(MakeJitConstant("USE_LOCAL_MEM", 1));
    }

    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    kernel.params.workGroups.global = {1, 1, 1};
    kernel.params.workGroups.local = {1, 1, 1};
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.params.arguments = GetArgsDesc(2, false, false);

    return {kd};
}

KernelsPriority GatherNonzeroKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool GatherNonzeroKernelRef::Validate(const Params& p, const optional_params& op) const {
    if (!KernelBaseOpenCL::Validate(p, op))
        return false;

    const auto& rp = static_cast<const gather_nonzero_params&>(p);

    return Tensor::SimpleLayout(rp.inputs[0].GetLayout());
}
}  // namespace kernel_selector
