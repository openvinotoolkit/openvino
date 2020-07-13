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


#include "lrn_kernel_across_channel_multiple_features.h"
#include <algorithm>

namespace kernel_selector {
ParamsKey LRNKernelAcrossChannelMultipleFeatures::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableLRNMode(LRNMode::ACROSS_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

static unsigned int GetOfmPerSimd(const lrn_params& params) {
    const auto& output = params.output;
    const auto local_size = params.localSize;

    if ((output.Feature().v % 8 == 0) && local_size > 4) {
        return 8;
    } else if ((output.Feature().v % 4 == 0) && local_size > 2) {
        return 4;
    } else if ((output.Feature().v % 2 == 0) && local_size > 1) {
        return 2;
    }

    return 1;
}

CommonDispatchData LRNKernelAcrossChannelMultipleFeatures::SetDefault(const lrn_params& params) const {
    CommonDispatchData runInfo = LRNKernelBase::SetDefault(params);
    const auto& input = params.inputs[0];

    unsigned int ofm_per_simd = GetOfmPerSimd(params);

    if (input.GetLayout() == DataLayout::bfyx || input.GetLayout() == DataLayout::b_fs_yx_fsv4) {
        const auto& out = params.output;
        const unsigned int alignment = out.X().v > 16 ? 32 : 16;

        runInfo.gws0 = Align(out.X().v, alignment);
        runInfo.gws1 = out.Y().v;
        runInfo.gws2 = (out.Feature().v * out.Batch().v) / ofm_per_simd;

        runInfo.lws0 = alignment;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;
    } else if (input.GetLayout() == DataLayout::yxfb) {
        runInfo.gws0 /= ofm_per_simd;
        runInfo.lws0 = std::min(std::max(runInfo.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (runInfo.gws0 % runInfo.lws0 != 0) {
            --runInfo.lws0;
        }
    }

    runInfo.efficiency = FORCE_PRIORITY_6;

    return runInfo;
}

bool LRNKernelAcrossChannelMultipleFeatures::Validate(const Params& p, const optional_params& o) const {
    if (!LRNKernelBase::Validate(p, o)) {
        return false;
    }

    const lrn_params& params = static_cast<const lrn_params&>(p);
    if (params.localSize > 32) {
        return false;
    }

    return true;
}

JitConstants LRNKernelAcrossChannelMultipleFeatures::GetJitConstants(const lrn_params& params, const DispatchData& kd) const {
    JitConstants jit = Parent::GetJitConstants(params, kd);
    const auto& input = params.inputs[0];
    const auto& input_dt = params.inputs[0].GetDType();
    const auto& output = params.output;

    unsigned int ofm_per_simd = GetOfmPerSimd(params);
    jit.AddConstant(MakeJitConstant("OFM_PER_SIMD", ofm_per_simd));

    if ((input.GetLayout() == DataLayout::bfyx || input.GetLayout() == DataLayout::b_fs_yx_fsv4) &&
        output.X().v <= 16) {
        jit.AddConstant(MakeJitConstant("FORCE_SIMD_16", 1));
    }

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"", {"batch_id", "feature_id + j", "y", "x"}, "lrn_result", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData LRNKernelAcrossChannelMultipleFeatures::GetKernelsData(const Params& params,
                                                                   const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_6);
}
}  // namespace kernel_selector
