// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <core/common/kernel_selector_utils.h>
#include "resample_kernel_ref.h"

#include <algorithm>
#include <vector>
#include <string>

namespace kernel_selector {

ParamsKey ResampleKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableReampleType(ResampleType::NEAREST_NEIGHBOR);
    k.EnableReampleType(ResampleType::CAFFE_BILINEAR_INTERP);
    k.EnableReampleType(ResampleType::BILINEAR_INTERP);
    k.EnableReampleType(ResampleType::CUBIC);
    k.EnableReampleType(ResampleType::LINEAR_ONNX);
    return k;
}

KernelsData ResampleKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

static size_t packing_factor(const resample_params& params) {
    // TODO Add support for only input packing
    bool in_out_8bit = (params.inputs[0].GetDType() == Datatype::UINT8 || params.inputs[0].GetDType() == Datatype::INT8) &&
                       (params.output.GetDType() == Datatype::UINT8 || params.output.GetDType() == Datatype::INT8);

    if (!in_out_8bit)
        return 1;

    auto get_layout_packing_factor = [](const DataLayout& layout) -> size_t {
        switch (layout) {
        case DataLayout::b_fs_yx_fsv16:
        case DataLayout::bs_fs_yx_bsv32_fsv16:
            return 16;
        case DataLayout::b_fs_yx_fsv4:
            return 4;
        default:
            break;
        }
        return 1;
    };

    size_t input_factor = get_layout_packing_factor(params.inputs[0].GetLayout());
    size_t output_factor = get_layout_packing_factor(params.output.GetLayout());

    if (input_factor % output_factor == 0 || output_factor % input_factor == 0)
        return std::min(input_factor, output_factor);
    return 1;
}

static bool use_packing(const resample_params& params) {
    if (params.resampleType != ResampleType::NEAREST_NEIGHBOR)
        return false;

    auto pack = packing_factor(params);
    if (pack == 1)
        return false;

    if (params.inputs[0].Feature().pad.before % pack != 0 || params.output.Feature().pad.before % pack != 0)
        return false;

    auto packed_work_items = params.output.X().v * params.output.Y().v * params.output.Z().v
        * CeilDiv(params.output.Feature().v, pack) * params.output.Batch().v;
    // TODO Loosen this requirement to minimum EUs needed to saturate cache bandwidth
    size_t max_work_items_per_eu = 32 * static_cast<size_t>(params.engineInfo.maxThreadsPerExecutionUnit);
    auto minimum_work_items = params.engineInfo.computeUnitsCount * max_work_items_per_eu;

    if (packed_work_items < minimum_work_items)
        return false;

    return true;
}

JitConstants ResampleKernelRef::GetJitConstants(const resample_params& params) const {
    JitConstants jit = ResampleKernelBase::GetJitConstants(params);

    if (use_packing(params)) {
        jit.AddConstant(MakeJitConstant("PACK_SIZE", packing_factor(params)));
        jit.AddConstant(MakeJitConstant("FEATURE_PACKED_MODE", "1"));
    }

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 4) {
            idx_order = {"batch", "OF_ID", "oy", "ox"};
        } else if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
            idx_order = {"batch", "OF_ID", "oz", "oy", "ox"};
        }

        FusedOpsConfiguration conf = {"", idx_order, "interp_val", GetAccumulatorType(params), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

ResampleKernelBase::DispatchData ResampleKernelRef::SetDefault(const resample_params& arg) const {
    auto dispatchData = Parent::SetDefault(arg);

    if (use_packing(arg)) {
        auto pack = packing_factor(arg);
        dispatchData.gws = { arg.output.X().v, arg.output.Y().v * arg.output.Z().v, CeilDiv(arg.output.Feature().v, pack) * arg.output.Batch().v };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo);
    }

    return dispatchData;
}

KernelsPriority ResampleKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
