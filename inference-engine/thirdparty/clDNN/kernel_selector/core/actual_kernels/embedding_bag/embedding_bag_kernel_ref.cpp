/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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
    CommonDispatchData runInfo;

    std::vector<size_t> global = { params.output.Batch().v,
                                   params.output.Feature().v,
                                   params.output.Y().v * params.output.X().v };

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}

KernelsData EmbeddingBagKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<embedding_bag_params>(params);
    embedding_bag_params& newParams = *static_cast<embedding_bag_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto runInfo = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel,
            runInfo,
            params.engineInfo,
            kernelName,
            jit,
            entry_point,
            "",
            false,
            false,
            (uint32_t)newParams.inputs.size());

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return { kd };
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

    k.EnableInputLayout(DataLayout::bfxy);

    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

bool EmbeddingBagKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::EMBEDDING_BAG ||
        o.GetType() != KernelType::EMBEDDING_BAG) {
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
