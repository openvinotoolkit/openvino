// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC_WITH_PADDING);
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey PoolingKernel_b_fs_yx_fsv16::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

size_t PoolingKernel_b_fs_yx_fsv16::GetBlockSize(const pooling_params& params) const {
    if (params.outputs[0].X().v > 4)
        return 8;
    else if (params.outputs[0].X().v > 1)
        return 2;
    else
        return 1;
}

size_t PoolingKernel_b_fs_yx_fsv16::GetSimdSize(const pooling_params& params) const {
    auto& out = params.outputs[0];
    // Use smaller simd size in case of global pooling and small channels count to have more threads
    if (out.X().v == 1 && out.Y().v == 1 && out.Feature().v < 64)
        return 8;
    else
        return 16;
}

PoolingKernelBase::DispatchData PoolingKernel_b_fs_yx_fsv16::SetDefault(const pooling_params& params) const {
    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);

    const auto& out = params.outputs[0];
    const size_t alignment = GetSimdSize(params);
    size_t x_block_size = GetBlockSize(params);
    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = CeilDiv(x, x_block_size) * y;
    dispatchData.gws[1] = Align(f, alignment);
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = alignment;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority PoolingKernel_b_fs_yx_fsv16::GetKernelsPriority(const Params& params) const {
    const auto& pooling_p = static_cast<const pooling_params&>(params);

    return pooling_p.outputs[0].Batch().v == 1 ? FORCE_PRIORITY_1 : FORCE_PRIORITY_7;
}

JitConstants PoolingKernel_b_fs_yx_fsv16::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    const size_t alignment = GetSimdSize(params);
    size_t x_block_size = GetBlockSize(params);
    auto input = params.inputs[0];
    auto output = params.outputs[0];
    auto jit = PoolingKernelBase::GetJitConstants(params, dispatchData);

    size_t input_line_size = params.poolStride.x * (x_block_size - 1) + params.poolSize.x;

    auto acc_type = GetAccumulatorType(params);
    jit.Merge(MakeTypeJitConstants(acc_type, "ACCUMULATOR"));

    auto can_preload_full_line = [&]() -> bool {
        const float max_reg_bytes = 128 * 32 * 0.95f;
        const size_t line_bytes = input_line_size * BytesPerElement(input.GetDType());
        const size_t acc_bytes = x_block_size * BytesPerElement(acc_type);

        const float req_bytes = static_cast<float>((line_bytes + acc_bytes) * alignment);

        return req_bytes < max_reg_bytes;
    };

    jit.AddConstant(MakeJitConstant("CAN_PRELOAD_FULL_LINE", can_preload_full_line()));
    jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", x_block_size));
    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", alignment));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(output.X().v, x_block_size)));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));

    if (params.outputs[0].Feature().v % 16 != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_vec = {"_VEC",
                                         {"b", "(f_block*FEATURE_SLICE_SIZE + f_val*SUB_GROUP_SIZE)", "y", "x"},
                                         "pool_result",
                                         input_dt,
                                         x_block_size,
                                         LoadType::LT_ALIGNED_READ,
                                         BoundaryCheck::ENABLED,
                                         IndexType::TENSOR_COORD,
                                         Tensor::DataChannelName::X};
        FusedOpsConfiguration conf_scalar = {"_SCALAR",
                                            {"b", "(f_block*FEATURE_SLICE_SIZE + f_val*SUB_GROUP_SIZE)", "y", "(x+i)"},
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

bool PoolingKernel_b_fs_yx_fsv16::Validate(const Params& p) const {
    if (!PoolingKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const pooling_params&>(p);
    const auto feature_block_size = 16;

    // Check that padding features doesn't miss-align the blocks
    if (params.inputs[0].Feature().pad.before % feature_block_size != 0 || params.outputs[0].Feature().pad.before % feature_block_size != 0)
        return false;

    return true;
}

KernelsData PoolingKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}
}  // namespace kernel_selector
