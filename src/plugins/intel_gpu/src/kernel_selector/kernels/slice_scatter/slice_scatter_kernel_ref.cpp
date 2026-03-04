// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice_scatter_kernel_ref.h"
#include "slice_scatter_kernel_utils.h"

#include <kernel_selector_utils.h>
#include <vector>

using kernel_selector::slice_scatter_utils::addJitConstantsForParam;
using kernel_selector::slice_scatter_utils::JIT_AXES_BUFF_SIZE_NAME;
using kernel_selector::slice_scatter_utils::MAX_SUPPORTED_DIM;

namespace kernel_selector {

KernelsData SliceScatterKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }
    KernelData kernel_data = KernelData::Default<slice_scatter_params>(params);
    slice_scatter_params& new_params = dynamic_cast<slice_scatter_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(new_params);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);
    auto slice_scatter_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, slice_scatter_specific_jit, entry_point);

    GetUpdateDispatchDataFunc(kernel_data);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, static_cast<int>(new_params.inputs.size()),
                     0, 1, new_params.has_dynamic_tensors());

    return {kernel_data};
}

KernelsPriority SliceScatterKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey SliceScatterKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    k.EnableDifferentTypes();
    return k;
}

bool SliceScatterKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SLICE_SCATTER) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const slice_scatter_params& params = dynamic_cast<const slice_scatter_params&>(p);
    if (params.inputs.empty())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.outputs[0].Dimentions() > MAX_SUPPORTED_DIM || params.inputs[0].Dimentions() > MAX_SUPPORTED_DIM)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

JitConstants SliceScatterKernelRef::GetJitConstants(const slice_scatter_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    // Define axes size as constant:
    if (params.compile_time_axes.empty()) {
        kernel_selector::DimensionAccessHelperJit dims(params.inputs.back());
        jit.AddConstant(MakeJitConstant(JIT_AXES_BUFF_SIZE_NAME,
                                        toVectorMulString({dims.b(), dims.f(), dims.x(), dims.y(), dims.z()})));
    } else {
        jit.AddConstant(MakeJitConstant(JIT_AXES_BUFF_SIZE_NAME, params.compile_time_axes.size()));
    }

    // Prepare axes, start and step params:
    const auto axes_decorator = [](std::string name, size_t i) {
        const std::string i_str = std::to_string(i);
        return name + "[" + i_str + "] < 0 ? INPUT0_DIMS + " + name + "[" + i_str + "] : " + name + "[" + i_str + "]";
    };
    addJitConstantsForParam(jit, "AXES", params.compile_time_axes, params.axes_data_type, axes_decorator);

    const auto default_decorator = [](std::string name, size_t i) {
        return name + "[" + std::to_string(i) + "]";
    };
    addJitConstantsForParam(jit, "START", params.compile_time_start, params.start_data_type, default_decorator);
    addJitConstantsForParam(jit, "STEP", params.compile_time_step, params.step_data_type, default_decorator);

    return jit;
}

CommonDispatchData SliceScatterKernelRef::SetDefault(const slice_scatter_params& params) const {
    CommonDispatchData dispatchData;
    // Dispatch over the updates tensor (INPUT1), same pattern as slice reads from output
    dispatchData.gws = { params.inputs[1].Batch().v, params.inputs[1].Feature().v,
            params.inputs[1].Z().v * params.inputs[1].Y().v * params.inputs[1].X().v };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws,
            params.engineInfo);

    return dispatchData;
}

void SliceScatterKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const slice_scatter_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

}  // namespace kernel_selector
