// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include"slice_kernel_ref.h"
#include <kernel_selector_utils.h>
#include <vector>

namespace {

void addJitConstantsForAttribute(kernel_selector::JitConstants &jit,
        const std::string &name, const std::vector<std::int32_t> &attribute) {
    using namespace kernel_selector;
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
    auto dispatch_data = SetDefault(new_params, options);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, options);
    auto slice_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, slice_specific_jit, entry_point);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo,
            kernelName, jit, entry_point);

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
    addJitConstantsForAttribute(jit, "SLICE_BEGIN", params.start);
    addJitConstantsForAttribute(jit, "SLICE_END", params.end);
    addJitConstantsForAttribute(jit, "SLICE_STEP", params.step);
    return jit;
}

CommonDispatchData SliceKernelRef::SetDefault(const slice_params &params,
        const optional_params&) const {
    CommonDispatchData dispatchData;
    dispatchData.gws = { params.outputs[0].Batch().v, params.outputs[0].Feature().v,
            params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws,
            params.engineInfo);

    return dispatchData;
}

} // namespace kernel_selector
