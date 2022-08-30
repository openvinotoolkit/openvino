// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyramid_roi_align_kernel_ref.h"
#include "kernel_selector_utils.h"

#include <vector>

namespace kernel_selector {
ParamsKey PyramidROIAlignKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);

    k.EnableBatching();
    k.EnableDifferentTypes();

    return k;
}

PyramidROIAlignKernelBase::DispatchData PyramidROIAlignKernelRef::SetDefault(const PyramidROIAlign_params& params) const {
    auto dispatchData = PyramidROIAlignKernelBase::SetDefault(params);
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::BATCH }};

    dispatchData.gws = {
        params.outputs[0].X().v * params.outputs[0].Y().v,
        params.outputs[0].Feature().v,
        params.outputs[0].Batch().v };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsData PyramidROIAlignKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority PyramidROIAlignKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
