// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "count_nonzero_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey CountNonzeroKernelRef::GetSupportedKey() const {
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
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

KernelsData CountNonzeroKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::COUNT_NONZERO);

    KernelData kd = KernelData::Default<count_nonzero_params>(params);
    count_nonzero_params& newParams = *static_cast<count_nonzero_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = MakeBaseParamsJitConstants(newParams);

    cldnn_jit.AddConstant(MakeJitConstant("OV_INPUT_RANK", newParams.ov_input_rank));
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    const auto& in = newParams.inputs[0];
    auto& kernel = kd.kernels[0];
    const auto& in_dims = in.GetDims();

    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    if (in_dims.size() == 4) {
        kernel.params.workGroups.global = {in_dims[0].v, in_dims[1].v, in_dims[2].v * in_dims[3].v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    } else if (in_dims.size() == 5) {
        kernel.params.workGroups.global = {in_dims[0].v, in_dims[1].v * in_dims[2].v, in_dims[3].v * in_dims[4].v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    } else {
        kernel.params.workGroups.global = {in_dims[0].v * in_dims[1].v, in_dims[2].v * in_dims[3].v, in_dims[4].v * in_dims[5].v};
        dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z, Tensor::DataChannelName::W},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    }

    kernel.params.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.params.workGroups.global,
                                                                   params.engineInfo,
                                                                   newParams.inputs[0].GetLayout(),
                                                                   newParams.outputs[0].GetLayout(),
                                                                   dims_by_gws);

    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.params.arguments = GetArgsDesc(1, false, false);

    return {kd};
}

KernelsPriority CountNonzeroKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool CountNonzeroKernelRef::Validate(const Params& p, const optional_params& op) const {
    if (!KernelBaseOpenCL::Validate(p, op))
        return false;

    const auto& rp = static_cast<const count_nonzero_params&>(p);

    return Tensor::SimpleLayout(rp.inputs[0].GetLayout());
}
}  // namespace kernel_selector
