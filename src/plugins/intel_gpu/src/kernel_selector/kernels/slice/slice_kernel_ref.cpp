// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include"slice_kernel_ref.h"
#include <kernel_selector_utils.h>
#include <vector>

namespace {
static constexpr size_t MAX_SUPPORTED_DIM = 5;
static constexpr char JIT_AXES_BUFF_SIZE_NAME[] = "AXES_BUFFER_SIZE";

// Generates macros:
// - name_BUFFER
// - name_VAL0, name_VAL1 ...
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
        // Dynamic param:
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
        // Static param:
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

KernelsData SliceKernelRef::GetKernelsData(const Params &params) const {
    if (!Validate(params)) {
        return {};
    }
    KernelData kernel_data = KernelData::Default<slice_params>(params);
    slice_params &new_params =
            dynamic_cast<slice_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(new_params);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);
    auto slice_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, slice_specific_jit, entry_point);

    GetUpdateDispatchDataFunc(kernel_data);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, static_cast<int>(new_params.inputs.size()),
                     0, 1, new_params.has_dynamic_tensors());

    return {kernel_data};
}

KernelsPriority SliceKernelRef::GetKernelsPriority(const Params&/*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey SliceKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::UINT8);
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

bool SliceKernelRef::Validate(const Params &p) const {
    if (p.GetType() != KernelType::SLICE) {
        return false;
    }

    const slice_params &params = dynamic_cast<const slice_params&>(p);
    if (params.inputs.empty())
        return false;

    if (params.outputs[0].Dimentions() > MAX_SUPPORTED_DIM || params.inputs[0].Dimentions() > MAX_SUPPORTED_DIM)
        return false;

    return true;
}

JitConstants SliceKernelRef::GetJitConstants(const slice_params& params) const {
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

CommonDispatchData SliceKernelRef::SetDefault(const slice_params &params) const {
    CommonDispatchData dispatchData;
    dispatchData.gws = { params.outputs[0].Batch().v, params.outputs[0].Feature().v,
            params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws,
            params.engineInfo);

    return dispatchData;
}

void SliceKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const slice_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

} // namespace kernel_selector
