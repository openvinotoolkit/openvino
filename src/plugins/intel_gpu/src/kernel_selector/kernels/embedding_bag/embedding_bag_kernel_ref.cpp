// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
JitConstants EmbeddingBagKernelRef::GetJitConstants(const embedding_bag_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    switch (params.type) {
    case EmbeddingBagType::PACKED_SUM:
        jit.AddConstant(MakeJitConstant("PACKED_SUM", 1));
        break;
    case EmbeddingBagType::OFFSETS_SUM:
        jit.AddConstant(MakeJitConstant("OFFSETS_SUM", 1));
        break;
    case EmbeddingBagType::SEGMENTS_SUM:
        jit.AddConstant(MakeJitConstant("SEGMENTS_SUM", 1));
        break;
    default:
        break;
    }
    if (params.default_index > -1)
        jit.AddConstant(MakeJitConstant("DEFAULT_INDEX", params.default_index));

    return jit;
}

CommonDispatchData EmbeddingBagKernelRef::SetDefault(const embedding_bag_params& params) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y }};

    dispatchData.gws = { params.outputs[0].Batch().v,
                         params.outputs[0].Feature().v,
                         params.outputs[0].Y().v * params.outputs[0].X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsData EmbeddingBagKernelRef::GetKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<embedding_bag_params>(params);
    embedding_bag_params& newParams = *static_cast<embedding_bag_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel,
            dispatchData,
            params.engineInfo,
            kernelName,
            jit,
            entry_point,
            "",
            false,
            false,
            (uint32_t)newParams.inputs.size());

    return { kd };
}

KernelsPriority EmbeddingBagKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey EmbeddingBagKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableInputDataType(Datatype::UINT32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);

    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

bool EmbeddingBagKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::EMBEDDING_BAG) {
        return false;
    }
    const embedding_bag_params& params = static_cast<const embedding_bag_params&>(p);

    auto checkIntType = [](Datatype dt) {
        if (dt != Datatype::INT32 && dt != Datatype::UINT32)
            return false;
        return true;
    };

    if (!checkIntType(params.inputs[1].GetDType()))
        return false;

    if (params.type == EmbeddingBagType::OFFSETS_SUM || params.type == EmbeddingBagType::SEGMENTS_SUM) {
        if (!checkIntType(params.inputs[2].GetDType()))
            return false;
    }

    return true;
}
}  // namespace kernel_selector
