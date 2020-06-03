// Copyright (c) 2018-2020 Intel Corporation
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


#include "pooling_kernel_gpu_b_fs_yx_fsv16.h"

namespace kernel_selector {
ParamsKey PoolingKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnablePoolType(PoolType::MAX);
    k.EnablePoolType(PoolType::AVG);
    k.EnablePoolType(PoolType::MAX_WITH_ARGMAX);
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC_WITH_PADDING);
    k.EnableDifferentTypes();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableDifferentTypes();
    return k;
}

size_t PoolingKernel_b_fs_yx_fsv16::GetBlockSize(const pooling_params& params) const {
    if (params.output.X().v > 4)
        return 8;
    else if (params.output.X().v > 1)
        return 2;
    else
        return 1;
}

PoolingKernelBase::DispatchData PoolingKernel_b_fs_yx_fsv16::SetDefault(const pooling_params& params) const {
    DispatchData kd = PoolingKernelBase::SetDefault(params);

    const auto& out = params.output;
    const size_t alignment = 16;
    size_t x_block_size = GetBlockSize(params);
    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    kd.gws0 = CeilDiv(x, x_block_size) * y;
    kd.gws1 = Align(f, alignment);
    kd.gws2 = b;

    kd.lws0 = 1;
    kd.lws1 = alignment;
    kd.lws2 = 1;

    kd.efficiency = FORCE_PRIORITY_2;

    return kd;
}

JitConstants PoolingKernel_b_fs_yx_fsv16::GetJitConstants(const pooling_params& params, DispatchData runInfo) const {
    const size_t alignment = 16;
    size_t x_block_size = GetBlockSize(params);
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = PoolingKernelBase::GetJitConstants(params, runInfo);

    size_t input_line_size = params.poolStride.x * (x_block_size - 1) + params.poolSize.x;

    jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", x_block_size));
    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", alignment));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(output.X().v, x_block_size)));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (params.output.Feature().v % 16 != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_vec = {"_VEC",
                                         {"b", "(f_block*16)", "y", "x"},
                                         "pool_result",
                                         input_dt,
                                         x_block_size,
                                         LoadType::LT_ALIGNED_READ,
                                         BoundaryCheck::ENABLED,
                                         IndexType::TENSOR_COORD,
                                         Tensor::DataChannelName::X};
        FusedOpsConfiguration conf_scalar = {"_SCALAR",
                                            {"b", "(f_block*16)", "y", "(x+i)"},
                                            "pool_result[i]",
                                            input_dt,
                                            1,
                                            LoadType::LT_ALIGNED_READ,
                                            BoundaryCheck::ENABLED,
                                            IndexType::TENSOR_COORD,
                                            Tensor::DataChannelName::X};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec, conf_scalar}));
    }

    return jit;
}

bool PoolingKernel_b_fs_yx_fsv16::Validate(const Params& p, const optional_params& o) const {
    if (!PoolingKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const pooling_params&>(p);
    const auto feature_block_size = 16;

    // Check that padding features doesn't miss-align the blocks
    if (params.inputs[0].Feature().pad.before % feature_block_size != 0 || params.output.Feature().pad.before % feature_block_size != 0)
        return false;

    return true;
}

KernelsData PoolingKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    const auto& pooling_p = static_cast<const pooling_params&>(params);
    if (pooling_p.output.Batch().v == 1)
        return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
    else
        return GetCommonKernelsData(params, options, FORCE_PRIORITY_7);
}
}  // namespace kernel_selector
