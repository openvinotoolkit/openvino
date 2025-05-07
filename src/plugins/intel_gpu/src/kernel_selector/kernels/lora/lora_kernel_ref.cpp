// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

enum KernelsTypes {
    BASE_KERNEL = 0,
    FUSED_OPS,
    TOTAL_KERNELS_NUM
};

static std::string GetKernelName(std::string base_name, size_t kernel_idx, const lora_params& params) {
    std::string kernel_name = base_name;

    if (params.lora_count > 1) {
        kernel_name += "_" + std::to_string(params.lora_count) + "_lora_fused";
    }

    if (kernel_idx == KernelsTypes::FUSED_OPS) {
        kernel_name += "_fused_ops";
    }

    return kernel_name;
}

ParamsKey LoRAKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

CommonDispatchData LoRAKernelRef::SetDefault(const lora_params& params, size_t kernel_idx) const {
    CommonDispatchData dispatchData;
    const auto& output = params.outputs[0];

    if (!output.is_dynamic()) {
        if (kernel_idx == KernelsTypes::BASE_KERNEL) {
            const auto& lora_rank_dim = params.inputs[3].Feature();
            size_t lora_rank = lora_rank_dim.is_dynamic ? 1 : lora_rank_dim.v;

            dispatchData.gws = { output.Batch().v * output.Feature().v,
                                 Align(output.Y().v * output.X().v, std::max(lora_rank, static_cast<size_t>(1))),
                                 1};

            dispatchData.lws = { 1, lora_rank, 1 };
        } else {
            dispatchData.gws = { output.Batch().v, output.Feature().v, output.Y().v * output.X().v };
            dispatchData.lws = GetOptimalLocalWorkGroupSizes({dispatchData.gws[0], dispatchData.gws[1], dispatchData.gws[2]}, params.engineInfo);
        }
    }

    return dispatchData;
}

JitConstants LoRAKernelRef::GetJitConstants(const lora_params& params, size_t kernel_idx) const {
    auto jit = Parent::GetJitConstants(params);

    if (kernel_idx == KernelsTypes::BASE_KERNEL) {
        jit.AddConstant(MakeJitConstant("BASE_KERNEL", 1));
        jit.AddConstant(MakeJitConstant("MAX_LORA_RANK", 256));
    } else {
        jit.AddConstant(MakeJitConstant("FUSED_OPS_KERNEL", 1));

        if (!params.fused_ops.empty()) {
            FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "output[output_idx]", params.outputs[0].GetDType()};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }

    return jit;
}

void LoRAKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const lora_params&>(params);
        OPENVINO_ASSERT(kd.kernels.size() == KernelsTypes::TOTAL_KERNELS_NUM, "[GPU] Invalid kernels size for update dispatch data func");

        auto dispatchDataBase = SetDefault(prim_params, KernelsTypes::BASE_KERNEL);
        kd.kernels[KernelsTypes::BASE_KERNEL].params.workGroups.global = dispatchDataBase.gws;
        kd.kernels[KernelsTypes::BASE_KERNEL].params.workGroups.local = dispatchDataBase.lws;
        kd.kernels[KernelsTypes::BASE_KERNEL].skip_execution = KernelData::SkipKernelExecution(prim_params);

        auto dispatchDataFusedOps = SetDefault(prim_params, KernelsTypes::FUSED_OPS);
        kd.kernels[KernelsTypes::FUSED_OPS].params.workGroups.global = dispatchDataFusedOps.gws;
        kd.kernels[KernelsTypes::FUSED_OPS].params.workGroups.local = dispatchDataFusedOps.lws;
        kd.kernels[KernelsTypes::FUSED_OPS].skip_execution = prim_params.fused_ops.empty();
    };
}

KernelsData LoRAKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const auto& prim_params = dynamic_cast<const lora_params&>(params);
    KernelData kd = KernelData::Default<lora_params>(params, KernelsTypes::TOTAL_KERNELS_NUM);

    GetUpdateDispatchDataFunc(kd);

    for (size_t kernel_idx = 0; kernel_idx < KernelsTypes::TOTAL_KERNELS_NUM; ++kernel_idx) {
        auto dispatchData = SetDefault(prim_params, kernel_idx);
        auto kernel_name = GetKernelName(kernelName, kernel_idx, prim_params);
        auto entry_point = GetEntryPoint(kernel_name, prim_params.layerID, params);
        auto cldnn_jit = GetJitConstants(prim_params, kernel_idx);
        auto jit = CreateJit(kernel_name, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[kernel_idx];

        if (kernel_idx == KernelsTypes::BASE_KERNEL) {
            size_t prim_inputs = prim_params.inputs.size();
            for (const auto& fd : prim_params.fused_ops) {
                prim_inputs -= fd.dep_size;
            }
            FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                             "", false, false, static_cast<int>(prim_inputs), 0, 1, prim_params.is_shape_agnostic);
        } else {
            FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                             "", false, false, 0, GetFusedPrimitiveInputsCount(params), 1, prim_params.is_shape_agnostic);
        }
    }

    return { kd };
}

KernelsPriority LoRAKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

}  // namespace kernel_selector
