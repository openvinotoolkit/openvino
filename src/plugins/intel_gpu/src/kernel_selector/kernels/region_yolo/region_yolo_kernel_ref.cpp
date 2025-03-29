// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "region_yolo_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
namespace {
RegionYoloKernelRef::DispatchData SetDefault(const region_yolo_params& params) {
    RegionYoloKernelRef::DispatchData dispatchData;

    const auto& input = params.inputs[0];
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::BATCH }};

    switch (input.GetLayout()) {
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::bfyx:
    case DataLayout::byxf: {
        uint32_t region_num = params.do_softmax ? params.num : params.mask_size;
        dispatchData.gws = {input.X().v * input.Y().v, region_num, input.Batch().v};
    } break;
    default:
        throw std::invalid_argument("Unsupported DataLayout");
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}
}  // namespace

ParamsKey RegionYoloKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants RegionYoloKernelRef::GetJitConstants(const region_yolo_params& ry) const {
    JitConstants jit = MakeBaseParamsJitConstants(ry);

    jit.AddConstants({MakeJitConstant("COORDS", ry.coords),
                      MakeJitConstant("CLASSES", ry.classes),
                      MakeJitConstant("NUM", ry.num),
                      MakeJitConstant("DO_SOFTMAX", ry.do_softmax),
                      MakeJitConstant("MASK_SIZE", ry.mask_size)});

    return jit;
}

bool RegionYoloKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::REGION_YOLO) {
        return false;
    }

    const region_yolo_params& params = static_cast<const region_yolo_params&>(p);
    const size_t expected_feature_size =
            params.do_softmax ? params.inputs[0].X().v * params.inputs[0].Y().v * params.inputs[0].Feature().v : params.inputs[0].Feature().v;

    if (expected_feature_size != params.outputs[0].Feature().v) {
        return false;
    }

    return true;
}

KernelsData RegionYoloKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const region_yolo_params& orgParams = static_cast<const region_yolo_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<region_yolo_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {kd};
}

KernelsPriority RegionYoloKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
