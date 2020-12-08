// Copyright (c) 2016-2020 Intel Corporation
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

#include "softmax_kernel_base.h"

namespace kernel_selector {
JitConstants SoftmaxKernelBase::GetJitConstants(const softmax_params& params,
                                                SoftmaxKernelBase::DispatchData dispatchData) const {
    JitConstants mem_consts = MakeBaseParamsJitConstants(params);

    mem_consts.AddConstants({MakeJitConstant("ALONG_" + toString(params.dim), "")});

    mem_consts.AddConstants({
        MakeJitConstant("ITEMS_NUM", dispatchData.itemsNum),
        MakeJitConstant("LWS", dispatchData.lws[0]),
        MakeJitConstant("GWS", dispatchData.gws[0]),
        MakeJitConstant("DATA_SETS_COUNT", dispatchData.dataSetsCount),
        MakeJitConstant("DATA_SET_SIZE", dispatchData.dataSetSize),
        MakeJitConstant("LEFTOVERS", dispatchData.leftovers),
    });

    return mem_consts;
}

SoftmaxKernelBase::DispatchData SoftmaxKernelBase::SetDefault(const softmax_params&,
                                                              const optional_params&) const {
    DispatchData dispatchData;

    dispatchData.gws[0] = 1;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    dispatchData.leftovers = 0;
    dispatchData.itemsNum = 0;
    dispatchData.normIndex = 0;
    dispatchData.dataSetsCount = 0;
    dispatchData.dataSetSize = 0;

    return dispatchData;
}

bool SoftmaxKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::SOFT_MAX || o.GetType() != KernelType::SOFT_MAX) {
        return false;
    }

    return true;
}

KernelsData SoftmaxKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const softmax_params& orgParams = static_cast<const softmax_params&>(params);
    KernelData kd = KernelData::Default<softmax_params>(params);

    auto dispatchData = SetDefault(orgParams, options);
    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = dispatchData.efficiency;

    return {kd};
}

bool SoftmaxKernelBaseBF::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    const softmax_params& params = static_cast<const softmax_params&>(p);
    const auto& input = params.inputs[0];

    if (!params.activations.empty()) {
        return false;
    }

    if (input.GetLayout() == DataLayout::bf || input.GetLayout() == DataLayout::fb) {
        return true;
    }

    switch (params.dim) {
        case SoftmaxDim::X:
            return input.Y().v == 1 && input.Z().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::Y:
            return input.X().v == 1 && input.Z().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::Z:
            return input.X().v == 1 && input.Y().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::FEATURE:
            return input.X().v == 1 && input.Y().v == 1 && input.Z().v == 1;
        default:
            return false;
    }
}

SoftmaxKernelBase::DispatchData SoftmaxKernelBaseBF::SetDefault(const softmax_params& params,
                                                                const optional_params& options) const {
    const auto& input = params.inputs[0];

    DispatchData dispatchData = Parent::SetDefault(params, options);

    auto flatten_input = input.FlattenFeatureAndSpatials();
    dispatchData.dataSetSize = flatten_input.Feature().v;
    dispatchData.dataSetsCount = input.Batch().v;

    return dispatchData;
}
}  // namespace kernel_selector
