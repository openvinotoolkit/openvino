// Copyright (c) 2016-2019 Intel Corporation
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
    constexpr size_t max_work_items_per_eu = 32 * 7;
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
    auto dispatch = Parent::SetDefault(arg);

    if (use_packing(arg)) {
        auto pack = packing_factor(arg);
        std::vector<size_t> global;
        std::vector<size_t> local;

        global = { arg.output.X().v, arg.output.Y().v * arg.output.Z().v, CeilDiv(arg.output.Feature().v, pack) * arg.output.Batch().v };
        local = GetOptimalLocalWorkGroupSizes(global, arg.engineInfo);

        dispatch.gws0 = global[0];
        dispatch.gws1 = global[1];
        dispatch.gws2 = global[2];

        dispatch.lws0 = local[0];
        dispatch.lws1 = local[1];
        dispatch.lws2 = local[2];
    }

    return dispatch;
}
}  // namespace kernel_selector
