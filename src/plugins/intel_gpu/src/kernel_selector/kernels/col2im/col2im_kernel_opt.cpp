// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey Col2ImKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData Col2ImKernelOpt::SetDefault(const col2im_params& params) const {
    CommonDispatchData dispatchData;

    auto input = params.inputs[0];
    const auto num_elements_for_block = input.Feature().v;
    const auto kernel_product = params.kernel_size.x * params.kernel_size.y;
    const auto num_channels = num_elements_for_block / kernel_product;

    dispatchData.gws = {num_channels, 1, params.outputs[0].Batch().v};
    dispatchData.lws = {1, 1, 1};

    return dispatchData;
}

KernelsData Col2ImKernelOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority Col2ImKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

JitConstants Col2ImKernelOpt::GetJitConstants(const col2im_params& params) const {
    auto jit = Parent::GetJitConstants(params);

    return jit;
}

}  // namespace kernel_selector
