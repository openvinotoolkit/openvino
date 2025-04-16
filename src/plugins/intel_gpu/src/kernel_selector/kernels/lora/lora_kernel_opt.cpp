// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_kernel_opt.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

enum KernelsTypes {
    FIRST_TOKEN_A = 0,
    FIRST_TOKEN_B,
    SECOND_TOKEN_A,
    SECOND_TOKEN_B,
    TOTAL_KERNELS_NUM
};

constexpr size_t max_workgroup_size = 512;
constexpr size_t gemm_a_sg_bk = 32;

} // namespace

static std::string GetKernelName(std::string base_name, KernelsTypes type, const lora_params& params) {
    std::string kernel_name = base_name;

    if (type == KernelsTypes::FIRST_TOKEN_A) {
        kernel_name += "_first_token_a";
    } else if (type == KernelsTypes::FIRST_TOKEN_B) {
        kernel_name += "_first_token_b";
    } else if (type == KernelsTypes::SECOND_TOKEN_A) {
        kernel_name += "_second_token_a";
    } else if (type == KernelsTypes::SECOND_TOKEN_B) {
        kernel_name += "_second_token_b";
    }

    return kernel_name;
}

bool LoRAKernelOpt::Validate(const Params& p) const {
    if (!LoRAKernelBase::Validate(p)) {
        return false;
    }
    const auto& prim_params = dynamic_cast<const lora_params&>(p);
    return !prim_params.is_ref_kernel;
}

ParamsKey LoRAKernelOpt::GetSupportedKey() const {
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

CommonDispatchData LoRAKernelOpt::SetDefault(const lora_params& params, size_t kernel_idx) const {
    CommonDispatchData dispatchData;

    if (!params.outputs[0].is_dynamic()) {
        if (kernel_idx == KernelsTypes::SECOND_TOKEN_A) {
            size_t lora_rank = params.inputs[3].Feature().v;
            size_t gemma_sgK = max_workgroup_size / lora_rank;
            size_t K = params.inputs[2].Feature().v;
            size_t gemma_wgs = CeilDiv(K, gemm_a_sg_bk * gemma_sgK);

            dispatchData.gws = { gemma_wgs, gemma_sgK, lora_rank };
            dispatchData.lws = { 1, gemma_sgK, lora_rank };
        } else if (kernel_idx == KernelsTypes::SECOND_TOKEN_B) {
            size_t output_state = params.inputs[4].Batch().v;

            dispatchData.gws = { RoundUp(output_state, max_workgroup_size), 1, 1 };
            dispatchData.lws = { max_workgroup_size, 1, 1 };
        } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_A) {
            dispatchData.gws = { 1, 1, 1 };
            dispatchData.lws = { 1, 1, 1 };
        } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_B) {
            dispatchData.gws = { 1, 1, 1 };
            dispatchData.lws = { 1, 1, 1 };
        }
    }

    return dispatchData;
}

JitConstants LoRAKernelOpt::GetJitConstants(const lora_params& params, size_t kernel_idx) const {
    auto jit = Parent::GetJitConstants(params);

    Datatype in_dtype = params.inputs[0].GetDType();
    size_t subgroup_size = in_dtype == Datatype::F16 ? 16 : 8;

    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("MAX_WORKGROUP_SIZE", max_workgroup_size));

    if (kernel_idx == KernelsTypes::FIRST_TOKEN_A) {
        jit.AddConstant(MakeJitConstant("FIRST_TOKEN_A", 1));
    } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_B) {
        jit.AddConstant(MakeJitConstant("FIRST_TOKEN_B", 1));
    } else if (kernel_idx == KernelsTypes::SECOND_TOKEN_A) {
        jit.AddConstant(MakeJitConstant("SECOND_TOKEN_A", 1));
    } else if (kernel_idx == KernelsTypes::SECOND_TOKEN_B) {
        jit.AddConstant(MakeJitConstant("SECOND_TOKEN_B", 1));
    }

    if (kernel_idx == KernelsTypes::SECOND_TOKEN_A || kernel_idx == KernelsTypes::SECOND_TOKEN_B) {
        DimensionAccessHelperJit state_a_dims(params.inputs[2]);
        jit.AddConstant(MakeJitConstant("K", state_a_dims.f()));

        jit.AddConstant(MakeJitConstant("GEMMA_SGK", "min(MAX_WORKGROUP_SIZE / LORA_RANK, MAX_GEMMA_SGK)"));

        jit.AddConstant(MakeJitConstant("GEMMA_SG_BK", gemm_a_sg_bk));

        jit.AddConstant(MakeJitConstant("GEMMB_SGN", "MAX_WORKGROUP_SIZE / SUBGROUP_SIZE"));

        jit.AddConstant(MakeJitConstant("GEMMB_PART_NUM", "CEIL_DIV(K, GEMMA_SG_BK * GEMMA_SGK)"));

        DimensionAccessHelperJit state_b_dims(params.inputs[4]);
        jit.AddConstant(MakeJitConstant("N", state_b_dims.b()));

        size_t max_gemma_sgk = in_dtype == Datatype::F16 ? 64 : 32;
        jit.AddConstant(MakeJitConstant("MAX_GEMMA_SGK", max_gemma_sgk));
        jit.AddConstant(MakeJitConstant("MAX_LORA_RANK", 256));
        jit.AddConstant(MakeJitConstant("MAX_GEMMA_SG_BK", 64));
    } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_A) {
        jit.AddConstant(MakeJitConstant("REG_M", 4));
        jit.AddConstant(MakeJitConstant("REG_N", 1));
        jit.AddConstant(MakeJitConstant("SG_M", "CEIL_DIV(SUBGROUP_SIZE, SG_N)"));
        jit.AddConstant(MakeJitConstant("SG_N", "LORA_RANK / SUBGROUP_SIZE / REG_N"));

        DimensionAccessHelperJit state_a_dims(params.inputs[2]);
        jit.AddConstant(MakeJitConstant("K", state_a_dims.f()));

        DimensionAccessHelperJit lora_input(params.inputs[1]);
        jit.AddConstant(MakeJitConstant("M", lora_input.b() + " * " + lora_input.f()));
    }

    return jit;
}

void LoRAKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const lora_params&>(params);
        OPENVINO_ASSERT(kd.kernels.size() == KernelsTypes::TOTAL_KERNELS_NUM, "[GPU] Invalid kernels size for update dispatch data func");

        auto dispatchDataFirstA = SetDefault(prim_params, KernelsTypes::FIRST_TOKEN_A);
        kd.kernels[KernelsTypes::FIRST_TOKEN_A].params.workGroups.global = dispatchDataFirstA.gws;
        kd.kernels[KernelsTypes::FIRST_TOKEN_A].params.workGroups.local = dispatchDataFirstA.lws;
        kd.kernels[KernelsTypes::FIRST_TOKEN_A].skip_execution = KernelData::SkipKernelExecution(prim_params);

        auto dispatchDataFirstB = SetDefault(prim_params, KernelsTypes::FIRST_TOKEN_B);
        kd.kernels[KernelsTypes::FIRST_TOKEN_B].params.workGroups.global = dispatchDataFirstB.gws;
        kd.kernels[KernelsTypes::FIRST_TOKEN_B].params.workGroups.local = dispatchDataFirstB.lws;
        kd.kernels[KernelsTypes::FIRST_TOKEN_B].skip_execution = KernelData::SkipKernelExecution(prim_params);

        auto dispatchDataSecondA = SetDefault(prim_params, KernelsTypes::SECOND_TOKEN_A);
        kd.kernels[KernelsTypes::SECOND_TOKEN_A].params.workGroups.global = dispatchDataSecondA.gws;
        kd.kernels[KernelsTypes::SECOND_TOKEN_A].params.workGroups.local = dispatchDataSecondA.lws;
        kd.kernels[KernelsTypes::SECOND_TOKEN_A].skip_execution = KernelData::SkipKernelExecution(prim_params);

        auto dispatchDataSecondB = SetDefault(prim_params, KernelsTypes::SECOND_TOKEN_B);
        kd.kernels[KernelsTypes::SECOND_TOKEN_B].params.workGroups.global = dispatchDataSecondB.gws;
        kd.kernels[KernelsTypes::SECOND_TOKEN_B].params.workGroups.local = dispatchDataSecondB.lws;
        kd.kernels[KernelsTypes::SECOND_TOKEN_B].skip_execution = KernelData::SkipKernelExecution(prim_params);

        kd.internalBuffers.clear();

        size_t input_state = prim_params.inputs[2].Feature().v;
        size_t lora_rank = prim_params.inputs[3].Feature().v;
        size_t output_a_size = CeilDiv(input_state, gemm_a_sg_bk * (max_workgroup_size / lora_rank));


        kd.internalBuffers.push_back(output_a_size * prim_params.inputs[0].ElementSize());
        kd.internalBufferDataType = prim_params.inputs[0].GetDType();
    };
}

KernelsData LoRAKernelOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const auto& prim_params = dynamic_cast<const lora_params&>(params);

    std::vector<KernelsTypes> kernels_type = {
        KernelsTypes::FIRST_TOKEN_A,
        KernelsTypes::FIRST_TOKEN_B,
        KernelsTypes::SECOND_TOKEN_A,
        KernelsTypes::SECOND_TOKEN_B
    };

    KernelData kd = KernelData::Default<lora_params>(params, kernels_type.size());

    GetUpdateDispatchDataFunc(kd);

    for (const auto& kernel_idx : kernels_type) {
        auto dispatchData = SetDefault(prim_params, kernel_idx);
        auto kernel_name = GetKernelName(kernelName, kernel_idx, prim_params);
        auto entry_point = GetEntryPoint(kernel_name, prim_params.layerID, params);
        auto cldnn_jit = GetJitConstants(prim_params, kernel_idx);
        auto jit = CreateJit(kernel_name, cldnn_jit, entry_point);


        auto& kernel = kd.kernels[kernel_idx];
        auto& args = kernel.params.arguments;

        if (kernel_idx == KernelsTypes::SECOND_TOKEN_A) {
            FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                             "", false, false, 2, 0, 0, prim_params.is_shape_agnostic);
            args.clear();
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT, 1});
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

            size_t default_bytes_count = 1;
            kd.internalBuffers.push_back(default_bytes_count);
            kd.internalBufferDataType = prim_params.inputs[0].GetDType();
        } else if (kernel_idx == KernelsTypes::SECOND_TOKEN_B) {
            FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                             "", false, false, 3, 0, 1, prim_params.is_shape_agnostic);
            args.clear();
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT, 0});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT, 3});
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_A) {
            FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                             "", false, false, static_cast<int>(prim_params.inputs.size()), 0, 1, prim_params.is_shape_agnostic);
        } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_B) {
            FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                             "", false, false, static_cast<int>(prim_params.inputs.size()), 0, 1, prim_params.is_shape_agnostic);
        }
    }

    return { kd };
}

KernelsPriority LoRAKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

}  // namespace kernel_selector
