// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_bf.h"
#include <algorithm>

namespace kernel_selector {
ParamsKey SoftmaxKernel_bf::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableSoftmaxDim(SoftmaxDim::X);  // in case that it can be flatten
    k.EnableSoftmaxDim(SoftmaxDim::Y);
    k.EnableSoftmaxDim(SoftmaxDim::FEATURE);
    k.EnableDifferentTypes();
    k.EnableBatching();
    return k;
}

SoftmaxKernel_bf::Parent::DispatchData SoftmaxKernel_bf::SetDefault(const softmax_params& params,
                                                                    const optional_params& optParams) const {
    auto dispatchData = Parent::SetDefault(params, optParams);

    // start with 1 thread per data set
    dispatchData.gws[0] = 1;
    dispatchData.gws[1] = dispatchData.dataSetsCount;
    dispatchData.itemsNum = dispatchData.dataSetSize;

    dispatchData.normIndex = 0;

    // We have two units of data per work item in current implementation.
    auto local_mem_per_wi = 2 * BytesPerElement(params.inputs[0].GetDType());
    // Combining device execution and local memory restrictions to compute maximum possible LWS.
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);

    dispatchData.lws[0] = 1;
    // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory
    // reads.
    while ((dispatchData.itemsNum > 32 || dispatchData.lws[0] < dispatchData.itemsNum) && (2 * dispatchData.lws[0] <= max_lws)) {
        dispatchData.lws[0] *= 2;
        dispatchData.itemsNum /= 2;
    }

    assert((dispatchData.itemsNum + 1) * dispatchData.lws[0] >= dispatchData.dataSetSize && "More than 'lws[0]' items per batch remains! Lws too small?");

    dispatchData.gws[0] = dispatchData.lws[0];
    dispatchData.leftovers = dispatchData.dataSetSize % dispatchData.lws[0];

    assert(dispatchData.itemsNum > 0 && dispatchData.lws[0] && dispatchData.gws[0] > 0);

    return dispatchData;
}

KernelsPriority SoftmaxKernel_bf::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_6;
}

KernelsData SoftmaxKernel_bf::GetKernelsData(const Params& params, const optional_params& optionalParams) const {
    return GetCommonKernelsData(params, optionalParams);
}

JitConstants SoftmaxKernel_bf::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_main = {"_MAIN",
                                           {"data_set_offset", "in_data_set_idx + i * workers_per_data_set", "0", "0"},
                                           "dequantized",
                                           input_dt};
        FusedOpsConfiguration conf_leftovers = {"_LEFTOVERS",
                                                {"data_set_offset", "workers_per_data_set * ITEMS_NUM + in_data_set_idx", "0", "0"},
                                                "dequantized",
                                                input_dt};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_main, conf_leftovers}));
    }

    return jit;
}
}  // namespace kernel_selector
