// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include"slice_kernel_ref.h"
#include <kernel_selector_utils.h>
#include <vector>

namespace {
static constexpr size_t MAX_SUPPORTED_DIM = 5;

std::string ovElementTypeToOCLStr(ov::element::Type_t type) {
#define CASE(TYPE, STR)     \
    case ov::element::TYPE: \
        return #STR;
    switch (type) {
        CASE(u64, ulong)
        CASE(i64, long)
        CASE(u32, uint)
        CASE(i32, int)
        CASE(u16, ushort)
        CASE(i16, short)
        CASE(u8, char)
        CASE(i8, uchar)

    default: {
        OPENVINO_ASSERT(false, "Unknown type!");
        return "unknown";
    }
    }

#undef CASE
}

// Generates macros:
// - name_BUFFER
// - name_DIM0, name_DIM1 ...
void addJitConstantsForParam(kernel_selector::JitConstants& jit,
                             const std::string& name,
                             const std::vector<std::int32_t>& compile_time_param,
                             const std::vector<std::int32_t>& compile_time_axes,
                             ov::element::Type_t type) {
    using namespace kernel_selector;

    const bool param_available_now = !compile_time_param.empty();
    const bool axes_available_now = !compile_time_axes.empty();

    if (param_available_now) {
        jit.AddConstant(MakeJitConstant(name + "_BUFFER", ""));
    } else {
        const std::string type_str = ovElementTypeToOCLStr(type);
        jit.AddConstant(
            MakeJitConstant(name + "_BUFFER", "__global const " + type_str + "* restrict " + name + "_buffer_ptr,"));
    }

    if (param_available_now && axes_available_now) {
        // Generate macros for compile_time_param available now and compile_time_axes available now.
        OPENVINO_ASSERT(compile_time_param.size() == compile_time_axes.size());
        for (size_t i = 0; i < compile_time_axes.size(); ++i) {
            jit.AddConstant(
                MakeJitConstant(name + "_DIM" + std::to_string(i), compile_time_param[compile_time_axes[i]]));
        }
    } else if (!param_available_now && axes_available_now) {
        // Generate macros for case where only compile_time_axes is available now.
        for (size_t i = 0; i < compile_time_axes.size(); ++i) {
            jit.AddConstant(MakeJitConstant(name + "_DIM" + std::to_string(i),
                                            name + "_buffer_ptr[" + std::to_string(compile_time_axes[i]) + "]"));
        }
    } else if (!param_available_now && !axes_available_now) {
        // Generate macros for case where both axes and param are available only in runtime.
        for (size_t i = 0; i < MAX_SUPPORTED_DIM; ++i) {
            const std::string axis_i = "SLICE_AXES_BUFFER_VAL(" + std::to_string(i) + ")";
            jit.AddConstant(MakeJitConstant(name + "_DIM" + std::to_string(i), name + "_buffer_ptr[" + axis_i + "]"));
        }
    } else {
        OPENVINO_ASSERT(
            false,
            "[SliceKernelRef]: Situation where param is available now and axes not - shouldn't be possible...");
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
    addJitConstantsForParam(jit,
                            "SLICE_BEGIN",
                            params.compile_time_start,
                            params.compile_time_axes,
                            params.start_data_type);
    addJitConstantsForParam(jit,
                            "SLICE_STEP",
                            params.compile_time_step,
                            params.compile_time_axes,
                            params.step_data_type);

    if (params.compile_time_axes.empty()) {
        const std::string type_str = ovElementTypeToOCLStr(params.axes_data_type);
        jit.AddConstant(MakeJitConstant("SLICE_AXES_BUFFER", "__global const " + type_str + "* restrict axes_ptr,"));
        jit.AddConstant(MakeJitConstant("SLICE_AXES_BUFFER_VAL(IDX)",
                                        "axes_ptr[IDX] < 0 ? INPUT0_DIMS + axes_ptr[IDX] : axes_ptr[IDX]"));
    } else {
        jit.AddConstant(MakeJitConstant("SLICE_AXES_BUFFER", ""));
    }
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
