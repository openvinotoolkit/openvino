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


#include "convolution_kernel_b_fs_yx_fsv4_int8.h"
#include <vector>
#include <utility>
#include <algorithm>

namespace kernel_selector {
constexpr size_t sub_group_size = 16;

ParamsKey ConvolutionKernel_b_fs_yx_fsv4_int8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableSubGroup();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv4_int8::SetDefault(const convolution_params& cp, int) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(cp);

    runInfo.efficiency = FORCE_PRIORITY_9;
    if (cp.output.X().v > 512 && cp.filterSize.x == 5 && cp.filterSize.y == 5)
        runInfo.efficiency = FORCE_PRIORITY_2;
    runInfo.gws0 = CeilDiv(cp.output.X().v, sub_group_size) / 2;
    runInfo.gws1 = cp.output.Y().v;
    runInfo.gws2 = sub_group_size;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = sub_group_size;

    return runInfo;
}

bool ConvolutionKernel_b_fs_yx_fsv4_int8::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);
    if (params.inputs[0].X().v % 64)
        return false;

    bool bFilterSize = (params.filterSize.x == 5 && params.filterSize.y == 5) ||
                       (params.filterSize.x == 3 && params.filterSize.y == 3 && (params.inputs[0].Feature().v % 4) == 0) ||
                       (params.filterSize.x == 1 && params.filterSize.y == 1);

    bool bStride = (params.stride.x == 1 && params.stride.y == 1);

    if (!bFilterSize || !bStride || (params.output.Feature().v % 4) != 0 || (params.output.Batch().v != 1)) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv4_int8::GetJitConstants(const convolution_params& params, const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws2));

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf0 = { "_0", {"batch", "FILTER_OFM_MAX * iter + ofm + 0", "idy", "idx"}, "res0", input_dt, 1 };
        FusedOpsConfiguration conf1 = { "_1", {"batch", "FILTER_OFM_MAX * iter + ofm + 1", "idy", "idx"}, "res1", input_dt, 1 };
        FusedOpsConfiguration conf2 = { "_2", {"batch", "FILTER_OFM_MAX * iter + ofm + 2", "idy", "idx"}, "res2", input_dt, 1 };
        FusedOpsConfiguration conf3 = { "_3", {"batch", "FILTER_OFM_MAX * iter + ofm + 3", "idy", "idx"}, "res3", input_dt, 1 };
        FusedOpsConfiguration conf4 = { "_4", {"batch", "FILTER_OFM_MAX * iter + ofm + 0", "idy", "idx"}, "res4", input_dt, 1 };
        FusedOpsConfiguration conf5 = { "_5", {"batch", "FILTER_OFM_MAX * iter + ofm + 1", "idy", "idx"}, "res5", input_dt, 1 };
        FusedOpsConfiguration conf6 = { "_6", {"batch", "FILTER_OFM_MAX * iter + ofm + 2", "idy", "idx"}, "res6", input_dt, 1 };
        FusedOpsConfiguration conf7 = { "_7", {"batch", "FILTER_OFM_MAX * iter + ofm + 3", "idy", "idx"}, "res7", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf0, conf1, conf2, conf3, conf4, conf5, conf6, conf7 }));
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv4_int8::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

}  // namespace kernel_selector
