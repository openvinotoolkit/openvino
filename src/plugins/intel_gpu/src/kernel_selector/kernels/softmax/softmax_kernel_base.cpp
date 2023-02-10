// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

SoftmaxKernelBase::DispatchData SoftmaxKernelBase::SetDefault(const softmax_params&) const {
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

    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const softmax_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(prim_params.inputs[0].PhysicalSizeInBytes());
        kd.internalBufferDataType = prim_params.inputs[0].GetDType();
    };

    auto dispatchData = SetDefault(orgParams);
    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);


    auto& kernel = kd.kernels[0];
    bool is_dynamic = orgParams.outputs[0].is_dynamic();

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     is_dynamic);

    if (is_dynamic) {
        auto& args = kernel.params.arguments;
        args.clear();
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(orgParams.inputs[0].PhysicalSizeInBytes());
        kd.internalBufferDataType = orgParams.inputs[0].GetDType();
    }

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
            return input.X().v == 1 && input.Z().v == 1 && (input.Feature().v == 1 || input.GetLayout() == DataLayout::bfyx);
        case SoftmaxDim::Z:
            return input.X().v == 1 && input.Y().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::FEATURE:
            return input.X().v == 1 && input.Y().v == 1 && input.Z().v == 1;
        default:
            return false;
    }
}

SoftmaxKernelBase::DispatchData SoftmaxKernelBaseBF::SetDefault(const softmax_params& params) const {
    const auto& input = params.inputs[0];

    DispatchData dispatchData = Parent::SetDefault(params);

    if (params.dim == SoftmaxDim::Y && input.Feature().v > 1 && input.GetLayout() == DataLayout::bfyx) {
        // Flatten BF for such case, X is expected to be 1
        OPENVINO_ASSERT(input.X().v == 1, "[GPU] SoftmaxKernelBaseBF: input.X() is expected to be 1 while actual value is ", input.X().v);
        dispatchData.dataSetSize = input.Y().v;
        dispatchData.dataSetsCount = input.Batch().v * input.Feature().v;
    } else {
        auto flatten_input = input.FlattenFeatureAndSpatials();
        dispatchData.dataSetSize = flatten_input.Feature().v;
        dispatchData.dataSetsCount = input.Batch().v;
    }

    return dispatchData;
}
}  // namespace kernel_selector
