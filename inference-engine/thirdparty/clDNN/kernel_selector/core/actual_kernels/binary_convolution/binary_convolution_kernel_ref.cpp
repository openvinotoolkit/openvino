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


#include "binary_convolution_kernel_ref.h"
#include <string>

namespace kernel_selector {

ParamsKey BinaryConvolutionKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BINARY);
    k.EnableInputWeightsType(WeightsType::BINARY);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableInputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    return k;
}

BinaryConvolutionKernelBase::DispatchData BinaryConvolutionKernelRef::SetDefault(
    const binary_convolution_params& params,
    int) const {
    DispatchData kd = BinaryConvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto b = out.Batch().v;
    auto f = out.Feature().v;
    auto y = out.Y().v;
    auto x = out.X().v;

    kd.gws0 = b;
    kd.gws1 = f;
    kd.gws2 = x * y;

    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = 1;

    kd.efficiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return kd;
}

JitConstants BinaryConvolutionKernelRef::GetJitConstants(const binary_convolution_params& params,
                                                         const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    int pad_physical_val = params.pad_value == -1.0f ? 0x00000000 : 0xFFFFFFFF;
    int leftovers_mask = (0xFFFFFFFF >> (32 - params.inputs[0].Feature().v % 32));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_NUM_PACKED", CeilDiv(params.inputs[0].Feature().v, 32)));
    jit.AddConstant(MakeJitConstant("FEATURE_PACK_SIZE", 32));
    jit.AddConstant(MakeJitConstant("OFM_BLOCK_SIZE", 32));
    jit.AddConstant(MakeJitConstant("EXCLUDE_PAD", params.pad_value == 0.0f));
    jit.AddConstant(MakeJitConstant("PAD_VALUE", pad_physical_val));
    jit.AddConstant(MakeJitConstant("LEFTOVERS", params.inputs[0].Feature().v % 32 != 0));
    jit.AddConstant(MakeJitConstant("LEFTOVERS_MASK", leftovers_mask));

    return jit;
}

KernelsData BinaryConvolutionKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

bool BinaryConvolutionKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (!BinaryConvolutionKernelBase::Validate(p, o) || !CovolutionBinaryCheckInput(p, o))
        return false;

    const auto& params = static_cast<const binary_convolution_params&>(p);

    if (!params.fused_ops.empty())
        return false;

    return true;
}

JitConstants BinaryConvolutionKernelRef::GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                                        const DispatchData& /*kd*/) const {
    JitConstants jit = {};

    auto input_dt = GetUnitType(params);
    FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "res", input_dt, 1 };
    jit.Merge(MakeFusedOpsJitConstants(params, {conf}));

    return jit;
}
}  // namespace kernel_selector
