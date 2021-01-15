// Copyright (c) 2019-2020 Intel Corporation
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

#include "resample_kernel_opt.h"
#include <vector>
#include <core/common/kernel_selector_utils.h>

namespace kernel_selector {

static constexpr size_t sub_group_size = 16;

size_t ResampleKernelOpt::GetOptimalBlockSize(const resample_params& params) const {
    std::vector<size_t> block_width = { 16, 8, 4, 2, 1 };
    for (auto& w : block_width)
        if (params.output.X().v % w == 0)
            return w;
    return 1;
}

ParamsKey ResampleKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableReampleType(ResampleType::BILINEAR_INTERP);
    k.EnableReampleType(ResampleType::NEAREST_NEIGHBOR);
    k.EnableReampleType(ResampleType::LINEAR_ONNX);
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

ResampleKernelBase::DispatchData ResampleKernelOpt::SetDefault(const kernel_selector::resample_params &arg) const {
    DispatchData dispatchData;
    const auto& out = arg.output;

    dispatchData.gws[0] = CeilDiv(out.X().v, GetOptimalBlockSize(arg)) * out.Y().v;
    dispatchData.gws[1] = Align(out.Feature().v, sub_group_size);
    dispatchData.gws[2] = arg.output.Batch().v;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = sub_group_size;
    dispatchData.lws[2] = 1;

    dispatchData.efficiency = FORCE_PRIORITY_3;

    return dispatchData;
}

bool ResampleKernelOpt::Validate(const Params& p, const optional_params& o) const {
    const resample_params& params = static_cast<const resample_params&>(p);

    if (!Parent::Validate(p, o))
        return false;

    if (p.GetType() != KernelType::RESAMPLE || o.GetType() != KernelType::RESAMPLE)
        return false;

    if (params.inputs.empty())
        return false;

    const auto& input = params.inputs[0];

    if (input.GetLayout() != DataLayout::fs_b_yx_fsv32 && input.GetLayout() != DataLayout::b_fs_yx_fsv16)
        return false;

    return true;
}

JitConstants ResampleKernelOpt::GetJitConstants(const resample_params &params) const {
    auto jit = Parent::GetJitConstants(params);

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", GetOptimalBlockSize(params)));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.output.X().v, GetOptimalBlockSize(params))));
    size_t vec_size = 0;
    if (params.inputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        vec_size = 2;
        jit.AddConstant(MakeJitConstant("FEATURE_SLICE_SIZE", 32));
    } else {
        vec_size = 1;
        jit.AddConstant(MakeJitConstant("FEATURE_SLICE_SIZE", 16));
    }
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order = {"b", "feature_block", "y", "(x + out_x)"};
        FusedOpsConfiguration conf = {"", idx_order, "res", GetAccumulatorType(params), vec_size, LoadType::LT_ALIGNED_READ};
        conf.SetVectorAxis(Tensor::DataChannelName::FEATURE);
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

Datatype ResampleKernelOpt::GetUnitType(const base_params& params) const {
    return params.inputs[0].GetDType();
}

KernelsData ResampleKernelOpt::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
