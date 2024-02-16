// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include"slice_kernel_ref.h"
#include <kernel_selector_utils.h>
#include <vector>

namespace {

void addJitConstantsForAttribute(kernel_selector::JitConstants &jit,
        const std::string &name, const std::vector<std::int32_t> &attribute,
        kernel_selector::base_params::ArgType arg_type ) {
    using namespace kernel_selector;

    if (arg_type == base_params::ArgType::Constant) {
        jit.AddConstant(MakeJitConstant(name + "_BUFFER", ""));
        jit.AddConstant(MakeJitConstant(name + "_BATCH", attribute[0]));
        jit.AddConstant(MakeJitConstant(name + "_FEATURE", attribute[1]));
        if (attribute.size() == 5) {  // BFZYX
            jit.AddConstant(MakeJitConstant(name + "_Z", attribute[2]));
            jit.AddConstant(MakeJitConstant(name + "_Y", attribute[3]));
            jit.AddConstant(MakeJitConstant(name + "_X", attribute[4]));
        } else {  // BFYX
            jit.AddConstant(MakeJitConstant(name + "_Y", attribute[2]));
            jit.AddConstant(MakeJitConstant(name + "_X", attribute[3]));
        }
    } else {
        jit.AddConstant(MakeJitConstant(name + "_BUFFER", "__global const ulong* " + name + "_buffer_ptr,"));
        jit.AddConstant(MakeJitConstant(name + "_BATCH", name + "_buffer_ptr[0]"));
        jit.AddConstant(MakeJitConstant(name + "_FEATURE", name + "_buffer_ptr[1]"));
        //jit.AddConstant(MakeJitConstant(name + "_Z", name + "_buffer_ptr[2]"));
        jit.AddConstant(MakeJitConstant(name + "_Y", name + "_buffer_ptr[2]"));
        jit.AddConstant(MakeJitConstant(name + "_X", name + "_buffer_ptr[3]"));
    }
}

} // anonymous namespace

namespace kernel_selector {

KernelsData SliceKernelRef::GetKernelsData(const Params &params,
        const optional_params &options) const {
    if (!Validate(params, options)) {
        return {};
    }
    KernelData kernel_data = KernelData::Default<slice_params>(params);
    slice_params &new_params =
            dynamic_cast<slice_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(new_params);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, options);
    auto slice_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, slice_specific_jit, entry_point);

    GetUpdateDispatchDataFunc(kernel_data);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, static_cast<int>(new_params.inputs.size()),
                     0, 1, new_params.has_dynamic_tensors());

    kernel_data.kernels[0].params.arguments;

    return {kernel_data};
}

KernelsPriority SliceKernelRef::GetKernelsPriority(const Params&/*params*/,
        const optional_params&/*options*/) const {
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

bool SliceKernelRef::Validate(const Params &p, const optional_params &o) const {
    if (p.GetType() != KernelType::SLICE || o.GetType() != KernelType::SLICE) {
        return false;
    }

    const slice_params &params = dynamic_cast<const slice_params&>(p);
    if (params.inputs.empty())
        return false;

    if (params.outputs[0].Dimentions() > 5 || params.inputs[0].Dimentions() > 5)
        return false;

    return true;
}

JitConstants SliceKernelRef::GetJitConstants(const slice_params &params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    addJitConstantsForAttribute(jit, "SLICE_BEGIN", params.start, params.start_arg_type);
    addJitConstantsForAttribute(jit, "SLICE_STEP", params.step, params.step_arg_type);
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
