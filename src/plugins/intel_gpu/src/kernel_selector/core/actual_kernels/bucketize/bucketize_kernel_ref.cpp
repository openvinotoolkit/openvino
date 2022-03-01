// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bucketize_kernel_ref.h"

#include <string>
#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey BucketizeKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

bool BucketizeKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::BUCKETIZE || o.GetType() != KernelType::BUCKETIZE) {
        return false;
    }

    const bucketize_params& params = static_cast<const bucketize_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 5)
        return false;

    return true;
}

CommonDispatchData BucketizeKernelRef::SetDefault(const bucketize_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {
        {Tensor::DataChannelName::BATCH},
        {Tensor::DataChannelName::FEATURE},
        {Tensor::DataChannelName::X, Tensor::DataChannelName::Y, Tensor::DataChannelName::Z}};

    dispatchData.gws = {params.outputs[0].Batch().v,
                        params.outputs[0].Feature().v,
                        params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v};
    dispatchData.lws =
        GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

JitConstants BucketizeKernelRef::GetJitConstants(const bucketize_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    return jit;
}

KernelsData BucketizeKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<bucketize_params>(params);
    bucketize_params& newParams = *static_cast<bucketize_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     DEFAULT,
                     false,
                     false,
                     2,
                     GetFusedPrimitiveInputsCount(params));

    return {kd};
}

KernelsPriority BucketizeKernelRef::GetKernelsPriority(const Params& /*params*/,
                                                       const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
