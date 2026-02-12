// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice_scatter_kernel_opt.h"
#include <kernel_selector_utils.h>
#include <vector>

namespace {
static constexpr size_t MAX_SUPPORTED_DIM = 5;
static constexpr char JIT_AXES_BUFF_SIZE_NAME[] = "AXES_BUFFER_SIZE";

void addJitConstantsForParam(kernel_selector::JitConstants& jit,
                             const std::string& name,
                             const std::vector<std::int64_t>& compile_time_param,
                             kernel_selector::Datatype type,
                             const std::function<std::string(std::string, size_t)>& dynamic_access_decorator) {
    using namespace kernel_selector;
    const std::string BUFF_CONST_NAME = name + "_BUFFER";
    const std::string BUFF_PTR_NAME = name + "_buffer_ptr";
    const auto jit_name_decorator = [](std::string name, size_t i) {
        return name + "_VAL" + std::to_string(i);
    };

    if (compile_time_param.empty()) {
        const std::string type_str = toCLType(type);
        jit.AddConstant(
            MakeJitConstant(BUFF_CONST_NAME, "__global const " + type_str + "* restrict " + BUFF_PTR_NAME + ","));
        for (size_t i = 0; i < MAX_SUPPORTED_DIM; ++i) {
            const std::string i_str = std::to_string(i);
            const std::string jit_name = jit_name_decorator(name, i);
            const std::string access_str = dynamic_access_decorator(BUFF_PTR_NAME, i);
            jit.AddConstant(
                MakeJitConstant(jit_name, i_str + " < " + JIT_AXES_BUFF_SIZE_NAME + " ? (" + access_str + ") : -1"));
        }
    } else {
        jit.AddConstant(MakeJitConstant(BUFF_CONST_NAME, ""));
        for (size_t i = 0; i < MAX_SUPPORTED_DIM; ++i) {
            const std::string jit_name = jit_name_decorator(name, i);
            const int64_t val = i < compile_time_param.size() ? compile_time_param[i] : -1;
            jit.AddConstant(MakeJitConstant(jit_name, val));
        }
    }
}

}  // anonymous namespace

namespace kernel_selector {

KernelsData SliceScatterKernelOpt::GetKernelsData(const Params& params) const {
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

KernelsPriority SliceScatterKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

ParamsKey SliceScatterKernelOpt::GetSupportedKey() const {
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

bool SliceScatterKernelOpt::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SLICE_SCATTER) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const slice_scatter_params& params = dynamic_cast<const slice_scatter_params&>(p);
    if (params.inputs.empty())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.outputs[0].Dimentions() > MAX_SUPPORTED_DIM || params.inputs[0].Dimentions() > MAX_SUPPORTED_DIM)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Opt kernel only valid when all steps are 1 (contiguous scatter)
    if (!params.compile_time_step.empty()) {
        for (const auto& s : params.compile_time_step) {
            if (s != 1)
                DO_NOT_USE_THIS_KERNEL(p.layerID);
        }
    } else {
        // Step is dynamic - can't guarantee step==1, fall back to ref
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // Inner dimension of updates must be large enough for vectorization
    const auto& updates = params.inputs[1];
    if (updates.X().v < VEC_SIZE)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

DeviceFeaturesKey SliceScatterKernelOpt::get_required_device_features_key(const Params& /*params*/) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_reqd_subgroup_size();
    return k;
}

JitConstants SliceScatterKernelOpt::GetJitConstants(const slice_scatter_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("SLICE_SCATTER_VEC_SIZE", VEC_SIZE));
    jit.AddConstant(MakeJitConstant("SIMD_SIZE", SIMD_SIZE));

    // Axes size
    if (params.compile_time_axes.empty()) {
        DimensionAccessHelperJit dims(params.inputs.back());
        jit.AddConstant(MakeJitConstant(JIT_AXES_BUFF_SIZE_NAME,
                                        toVectorMulString({dims.b(), dims.f(), dims.x(), dims.y(), dims.z()})));
    } else {
        jit.AddConstant(MakeJitConstant(JIT_AXES_BUFF_SIZE_NAME, params.compile_time_axes.size()));
    }

    const auto axes_decorator = [](std::string name, size_t i) {
        const std::string i_str = std::to_string(i);
        return name + "[" + i_str + "] < 0 ? INPUT0_DIMS + " + name + "[" + i_str + "] : " + name + "[" + i_str + "]";
    };
    addJitConstantsForParam(jit, "AXES", params.compile_time_axes, params.axes_data_type, axes_decorator);

    const auto default_decorator = [](std::string name, size_t i) {
        return name + "[" + std::to_string(i) + "]";
    };
    addJitConstantsForParam(jit, "START", params.compile_time_start, params.start_data_type, default_decorator);

    // Step is always compile-time 1 for opt kernel, but we still provide STEP_BUFFER for CL compatibility
    addJitConstantsForParam(jit, "STEP", params.compile_time_step, params.step_data_type, default_decorator);

    return jit;
}

CommonDispatchData SliceScatterKernelOpt::SetDefault(const slice_scatter_params& params) const {
    CommonDispatchData dispatchData;
    const auto& updates = params.inputs[1];

    // Dispatch: dim0=batch, dim1=feature, dim2=ceil(Z*Y*X / VEC_SIZE)
    const size_t total_spatial = updates.Z().v * updates.Y().v * updates.X().v;
    const size_t spatial_work_items = (total_spatial + VEC_SIZE - 1) / VEC_SIZE;

    dispatchData.gws = { updates.Batch().v, updates.Feature().v, spatial_work_items };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

void SliceScatterKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
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
