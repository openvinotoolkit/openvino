// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_kernel_base.h"
#include <algorithm>

namespace kernel_selector {
bool PoolingKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::POOLING) {
        return false;
    }

    auto& params = dynamic_cast<const pooling_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 5)
        return false;

    return true;
}

Datatype PoolingKernelBase::GetAccumulatorType(const pooling_params& params) const {
    const auto& input_dt = params.inputs[0].GetDType();
    const auto& pool_type = params.poolType;

    if (pool_type == PoolType::MAX) {
        return input_dt;
    } else {
        switch (input_dt) {
            case Datatype::F32: return Datatype::F32;
            case Datatype::F16: return Datatype::F32;
            case Datatype::INT8: return Datatype::INT32;
            case Datatype::UINT8: return Datatype::INT32;
            default: return Datatype::F32;
        }
    }
}

Datatype PoolingKernelBase::GetActivationType(const pooling_params& params) const {
    if (params.outputs[0].GetDType() == Datatype::F16)
        return Datatype::F16;
    else
        return Datatype::F32;
}


JitConstants PoolingKernelBase::GetJitConstants(const pooling_params& pp, PoolingKernelBase::DispatchData dispatchData) const {
    JitConstants mem_consts = MakeBaseParamsJitConstants(pp);

    mem_consts.AddConstants({
        MakeJitConstant("POOL", pp.poolSize),
        MakeJitConstant("STRIDE", pp.poolStride),
        MakeJitConstant("PADDING", pp.poolPad),
        MakeJitConstant(toString(pp.poolType) + "_POOLING", 1),
        MakeJitConstant(toString(pp.divMode) + "_KERNEL_DIVIDER", 1),
    });

    if (pp.maxPoolOpset8Features) {
        mem_consts.AddConstants({MakeJitConstant("DILATION", pp.poolDilation)});

        if (pp.poolAxis != 0) {
            size_t indices_upper_bound = 1;
            const auto& dims = pp.inputs[0].GetDims();
            for (auto d = dims.crbegin() + pp.poolAxis; d != dims.crend(); ++d) {
                indices_upper_bound *= d->v;
            }
            if (indices_upper_bound != 0 && indices_upper_bound != 1) {
                mem_consts.AddConstants({MakeJitConstant("INDICES_UPPER_BOUND", indices_upper_bound)});
            }
        }

        mem_consts.Merge(MakeTypeJitConstants(pp.poolIndexElementType, "SELECTED_INDICES"));
    }

    if (dispatchData.needsBoundary) {
        mem_consts.AddConstant(MakeJitConstant("CHECK_BOUNDARY", 1));
    }

    if (EnableRound(pp)) {
        mem_consts.AddConstant(MakeJitConstant("ENABLE_ROUND", 1));
    }

    return mem_consts;
}

// Checks if we need boundary checking in kernel.
bool PoolingKernelBase::NeedsBoundaryCheck(const pooling_params& pp) const {
    const auto& input = pp.inputs[0];
    const auto& output = pp.outputs[0];

    if (pp.poolPad.x != 0 || pp.poolPad.y != 0 || pp.poolPad.z != 0) {
        return true;
    } else if (pp.poolDilation.x > 1 || pp.poolDilation.y > 1 || pp.poolDilation.z > 1) {
        return true;
    } else if ((((input.X().v - pp.poolSize.x) / pp.poolStride.x) + 1) < output.X().v ||
               (((input.Y().v - pp.poolSize.y) / pp.poolStride.y) + 1) < output.Y().v ||
               (((input.Z().v - pp.poolSize.z) / pp.poolStride.z) + 1) < output.Z().v) {
        return true;
    }

    if (input.X().v < pp.poolSize.x || input.Y().v < pp.poolSize.y || input.Z().v < pp.poolSize.z) {
        return true;
    }

    if (pp.poolSize.x < 3 || pp.poolSize.y < 3) {
        return true;
    }

    auto mod_x = (input.X().v - pp.poolSize.x) % pp.poolStride.x;
    auto mod_y = (input.Y().v - pp.poolSize.y) % pp.poolStride.y;
    auto mod_z = (input.Z().v - pp.poolSize.z) % pp.poolStride.z;

    return mod_x || mod_y || mod_z;
}

bool PoolingKernelBase::EnableRound(const kernel_selector::pooling_params& params) const {
    bool has_fused_quantize_to_int8 = false;
    for (auto& op : params.fused_ops) {
        if (op.GetType() == FusedOpType::QUANTIZE &&
            (op.output_tensor.GetDType() == Datatype::INT8 || op.output_tensor.GetDType() == Datatype::UINT8)) {
            has_fused_quantize_to_int8 = true;
        }
    }

    if (!has_fused_quantize_to_int8 &&
        (params.outputs[0].GetDType() == Datatype::INT8 || params.outputs[0].GetDType() == Datatype::UINT8) &&
        params.poolType == PoolType::AVG) {
        return true;
    }

    return false;
}

PoolingKernelBase::DispatchData PoolingKernelBase::SetDefault(const pooling_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;

    if (output.GetLayout() == DataLayout::bfyx || output.GetLayout() == DataLayout::b_fs_yx_fsv4 ||
        output.GetLayout() == DataLayout::byxf ||
        output.GetLayout() == DataLayout::bfzyx || output.GetLayout() == DataLayout::b_fs_zyx_fsv16 ||
        output.GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16 ||
        output.GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv32 ||
        output.GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32) {
        // Determine global work sizes.
        dispatchData.gws[0] = Align(output.X().v, 32);                // X
        dispatchData.gws[1] = output.Y().v * output.Z().v;            // Y, Z
        dispatchData.gws[2] = output.Batch().v * output.Feature().v;  // B, F

        // Find largest positive local work size that is divider for global work size.
        dispatchData.lws[0] = 32;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    } else if (output.GetLayout() == DataLayout::b_fs_yx_fsv32 || output.GetLayout() == DataLayout::b_fs_zyx_fsv32) {
        dispatchData.gws[0] = 32;
        dispatchData.gws[1] = output.Y().v * output.X().v * output.Z().v;
        dispatchData.gws[2] = output.Batch().v * CeilDiv(output.Feature().v, 32);

        dispatchData.lws[0] = 32;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    } else {
        // Determine global work sizes.
        dispatchData.gws[0] = output.Batch().v * output.Feature().v;  // B, F
        dispatchData.gws[1] = output.X().v;                           // X
        dispatchData.gws[2] = output.Y().v * output.Z().v;            // Y * Z

        dispatchData.lws[0] = std::min(std::max(dispatchData.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
        while (dispatchData.gws[0] % dispatchData.lws[0] != 0) {
            --dispatchData.lws[0];
        }
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    dispatchData.needsBoundary = NeedsBoundaryCheck(params);

    return dispatchData;
}

KernelsData PoolingKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const pooling_params& orgParams = static_cast<const pooling_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<pooling_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, EXE_MODE_DEFAULT, false, false, 1,
                     GetFusedPrimitiveInputsCount(params));

    if (orgParams.maxPoolOpset8Features) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
    }

    return {kd};
}
}  // namespace kernel_selector
