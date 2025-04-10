// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey ReshapeKernelRef::GetSupportedKey() const {
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
    return k;
}

KernelsData ReshapeKernelRef::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::RESHAPE);

    KernelData kd = KernelData::Default<reshape_params>(params);
    reshape_params& newParams = *static_cast<reshape_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = MakeBaseParamsJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    const auto& in = newParams.inputs[0];
    auto& kernel = kd.kernels[0];
    size_t gws0 = 1;
    size_t gws1 = 1;
    size_t gws2 = 1;
    const auto& in_dims = in.GetDims();

    if (in_dims.size() >= 1)
        gws0 = in_dims[0].v;
    if (in_dims.size() >= 2)
        gws1 = in_dims[1].v;
    for (size_t i = 2; i < in_dims.size(); ++i) {
        gws2 *= in_dims[i].v;
    }

    kernel.params.workGroups.global = {gws0, gws1, gws2};
    kernel.params.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.params.workGroups.global, params.engineInfo);
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);
    kernel.params.arguments = GetArgsDesc(1, false, false);

    return {kd};
}

KernelsPriority ReshapeKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool ReshapeKernelRef::Validate(const Params& p) const {
    if (!KernelBaseOpenCL::Validate(p))
        return false;

    const auto& rp = static_cast<const reshape_params&>(p);

    return Tensor::SimpleLayout(rp.inputs[0].GetLayout());
}
}  // namespace kernel_selector
