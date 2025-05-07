// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_kernel_opt.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

enum KernelsTypes {
    FIRST_TOKEN_A_SMALL = 0,
    FIRST_TOKEN_A_MEDIUM,
    FIRST_TOKEN_A_LARGE,
    FIRST_TOKEN_B_MEDIUM,
    FIRST_TOKEN_B_LARGE,
    SECOND_TOKEN_A,
    SECOND_TOKEN_B,
    FUSED_OPS,
    TOTAL_KERNELS_NUM
};

constexpr size_t gemm_a_sg_bk = 32;

} // namespace

static std::string GetKernelName(std::string base_name, size_t kernel_idx, const lora_params& params) {
    std::string kernel_name = base_name;

    if (kernel_idx == KernelsTypes::FIRST_TOKEN_A_SMALL) {
        kernel_name += "_first_token_a_small";
    } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_A_MEDIUM) {
        kernel_name += "_first_token_a_medium";
    } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_A_LARGE) {
        kernel_name += "_first_token_a_large";
    } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_B_MEDIUM) {
        kernel_name += "_first_token_b_medium";
    } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_B_LARGE) {
        kernel_name += "_first_token_b_large";
    } else if (kernel_idx == KernelsTypes::SECOND_TOKEN_A) {
        kernel_name += "_second_token_a";
    } else if (kernel_idx == KernelsTypes::SECOND_TOKEN_B) {
        kernel_name += "_second_token_b";
    } else if (kernel_idx == KernelsTypes::FUSED_OPS) {
        kernel_name += "_fused_ops";
    }

    return kernel_name;
}

bool LoRAKernelOpt::Validate(const Params& p) const {
    if (!LoRAKernelBase::Validate(p)) {
        return false;
    }
    const auto& prim_params = dynamic_cast<const lora_params&>(p);
    return !prim_params.is_ref_kernel && prim_params.lora_count == 1;
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

DeviceFeaturesKey LoRAKernelOpt::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_blocked_read_write();
    k.requires_blocked_read_write_short();
    k.requires_subgroups();
    k.requires_subgroup_broadcast();

    return k;
}

LoRAKernelOpt::LoRATuningData LoRAKernelOpt::GetTuningParams(const lora_params& params, size_t kernel_idx) const {
    LoRATuningData tuning_data;

    const auto& in_dtype = params.inputs[0].GetDType();
    tuning_data.subgroup_size = in_dtype == Datatype::F16 ? 16 : 8;
    tuning_data.max_gemma_sgk = in_dtype == Datatype::F16 ? 64 : 32;

    switch (kernel_idx) {
    case KernelsTypes::FIRST_TOKEN_A_SMALL:
        tuning_data.regM = 4;
        tuning_data.regN = 1;
        break;
    case KernelsTypes::FIRST_TOKEN_A_MEDIUM:
        tuning_data.regM = 8;
        tuning_data.regN = 2;
        break;
    case KernelsTypes::FIRST_TOKEN_A_LARGE:
        tuning_data.regM = 16;
        tuning_data.regN = 2;
        break;
    case KernelsTypes::FIRST_TOKEN_B_MEDIUM:
        tuning_data.regM = 8;
        tuning_data.regN = 2;
        tuning_data.sgM = 16;
        tuning_data.sgN = 4;
        break;
    case KernelsTypes::FIRST_TOKEN_B_LARGE:
        tuning_data.regM = 16;
        tuning_data.regN = 2;
        tuning_data.sgM = 8;
        tuning_data.sgN = 4;
        break;
    default:
        break;
    }

    const auto& alpha_input = params.inputs[3];
    if (!alpha_input.is_dynamic() && kernel_idx <= KernelsTypes::FIRST_TOKEN_A_LARGE) {
        size_t lora_rank = alpha_input.Feature().v;

        tuning_data.sgN = lora_rank / tuning_data.subgroup_size / tuning_data.regN;
        if (tuning_data.sgN != 0) {
            tuning_data.sgM = CeilDiv(tuning_data.subgroup_size, tuning_data.sgN);
        }
    }

    return tuning_data;
}

std::pair<size_t, size_t> LoRAKernelOpt::GetSuitableKernels(const lora_params& params) const {
    std::pair<size_t, size_t> suitable_kernels;

    const auto& lora_input = params.inputs[1];
    OPENVINO_ASSERT(!lora_input.is_dynamic(), "[GPU] Unable to find suitable kernel for dynamic input");

    bool is_first_token = lora_input.Batch().v * lora_input.Feature().v > 1;
    if (is_first_token) {
        size_t batch = lora_input.Batch().v * lora_input.Feature().v;
        size_t lora_rank = params.inputs[3].Feature().v;

        if (lora_rank == 128 || lora_rank == 256) {
            suitable_kernels.first = KernelsTypes::FIRST_TOKEN_A_LARGE;
        } else if (lora_rank == 64) {
            suitable_kernels.first = KernelsTypes::FIRST_TOKEN_A_MEDIUM;
        } else {
            suitable_kernels.first = KernelsTypes::FIRST_TOKEN_A_SMALL;
        }

        if (batch < 256) {
            suitable_kernels.second = KernelsTypes::FIRST_TOKEN_B_MEDIUM;

            size_t max_workgroup_size = params.engineInfo.maxWorkGroupSize;
            const auto& tuned_params = GetTuningParams(params, KernelsTypes::FIRST_TOKEN_B_MEDIUM);
            if (tuned_params.sgM * tuned_params.sgN * tuned_params.subgroup_size > max_workgroup_size) {
                suitable_kernels.second = KernelsTypes::FIRST_TOKEN_B_LARGE;
            }
        } else {
            suitable_kernels.second = KernelsTypes::FIRST_TOKEN_B_LARGE;
        }
    } else {
        suitable_kernels = { KernelsTypes::SECOND_TOKEN_A, KernelsTypes::SECOND_TOKEN_B };
    }

    return suitable_kernels;
}

CommonDispatchData LoRAKernelOpt::SetDefault(const lora_params& params, size_t kernel_idx) const {
    CommonDispatchData dispatchData;
    size_t max_workgroup_size = params.engineInfo.maxWorkGroupSize;

    if (!params.outputs[0].is_dynamic()) {
        const auto& tuning_params = GetTuningParams(params, kernel_idx);

        if (kernel_idx == KernelsTypes::SECOND_TOKEN_A) {
            size_t lora_rank = params.inputs[3].Feature().v;
            size_t gemma_sgK = max_workgroup_size / std::max(lora_rank, static_cast<size_t>(1));
            size_t K = params.inputs[2].Feature().v;
            size_t gemma_wgs = CeilDiv(K, gemm_a_sg_bk * gemma_sgK);

            dispatchData.gws = { gemma_wgs, gemma_sgK, lora_rank };
            dispatchData.lws = { 1, gemma_sgK, lora_rank };
        } else if (kernel_idx == KernelsTypes::SECOND_TOKEN_B) {
            size_t output_state = params.inputs[4].Batch().v;

            dispatchData.gws = { RoundUp(output_state, max_workgroup_size), 1, 1 };
            dispatchData.lws = { max_workgroup_size, 1, 1 };
        } else if (kernel_idx == KernelsTypes::FUSED_OPS) {
            const auto& output = params.outputs[0];
            auto out_layout = output.GetLayout();

            std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                             { Tensor::DataChannelName::FEATURE },
                                                                             { Tensor::DataChannelName::Y, Tensor::DataChannelName::X }};

            dispatchData.gws = { output.Batch().v, output.Feature().v, output.Y().v * output.X().v };
            dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, out_layout, out_layout, dims_by_gws);
        } else {
            const auto& lora_input = params.inputs[1];
            size_t batch = lora_input.Batch().v * lora_input.Feature().v;

            size_t N = 0;
            if (kernel_idx <= FIRST_TOKEN_A_LARGE) {
                N = params.inputs[3].Feature().v;
            } else {
                N = params.inputs[4].Batch().v;
            }

            size_t BM = tuning_params.regM * tuning_params.sgM;
            size_t BN = tuning_params.regN * tuning_params.sgN * tuning_params.subgroup_size;

            if (tuning_params.sgN != 0) {
                dispatchData.gws = { RoundUp(batch, BM) / tuning_params.regM , RoundUp(N, BN) / tuning_params.regN, 1 };
                dispatchData.lws = { tuning_params.sgM, tuning_params.sgN * tuning_params.subgroup_size, 1 };
            }
        }
    }

    return dispatchData;
}

std::string LoRAKernelOpt::GenerateBlockRead(Datatype dtype, std::string input) const {
    std::string res = dtype == Datatype::F16 ? "intel_sub_group_block_read_us((const __global ushort*)("
                                             : "intel_sub_group_block_read((const __global uint*)(";
    res += input + "))";
    return res;
}

std::string LoRAKernelOpt::GenerateBlockWrite(Datatype dtype, std::string dst, std::string src) const {
    std::string res = "";
    if (dtype == Datatype::F16) {
        res = "intel_sub_group_block_write_us((__global ushort*)(" + dst + "), as_short(" + src + "));";
    } else {
        res = "intel_sub_group_block_write((__global uint*)(" + dst + "), as_int(" + src + "));";
    }
    return res;
}

std::string LoRAKernelOpt::GenerateBroadcast(Datatype dtype, std::string input) const {
    std::string res = dtype == Datatype::F16 ? "intel_sub_group_broadcast("
                                             : "sub_group_broadcast(";
    res += input + ")";
    return res;
}

std::string LoRAKernelOpt::GenerateMatMulCode(size_t M, size_t N, Datatype dtype, bool is_A_kernel) const {
    std::string res = "";
    std::string int_type = dtype == Datatype::F16 ? "ushort" : "uint";
    std::string input_type = is_A_kernel ? "INPUT1_TYPE" : "ACCUMULATOR_TYPE";

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            res += "ACCUMULATOR_TYPE sum_" + std::to_string(m) + "_" + std::to_string(n) + " = 0;";
        }
    }

    res += "for (int i = 0; i < K; i += SUBGROUP_SIZE) {";

    for (size_t m = 0; m < M; ++m) {
        res += int_type + " input_" + std::to_string(m) + " = " + GenerateBlockRead(dtype, "ptrA + " + std::to_string(m) + " * K") + ";";
    }

    res += "for (int kk = 0; kk < SUBGROUP_SIZE; kk++) {";

    for (size_t n = 0; n < N; ++n) {
        res += "ACCUMULATOR_TYPE bb_" + std::to_string(n) + " = "
            + "TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(" + GenerateBlockRead(dtype, "ptrB + " + std::to_string(n) + " * SUBGROUP_SIZE") + "));";
    }

    for (size_t m = 0; m < M; ++m) {
        res += "ACCUMULATOR_TYPE aa_" + std::to_string(m) + " = "
            + "TO_ACCUMULATOR_TYPE(AS_" + input_type + "(" + GenerateBroadcast(dtype, "input_" + std::to_string(m) + ", kk") + "));";
    }

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            std::string sum_var = "sum_" + std::to_string(m) + "_" + std::to_string(n);
            res += sum_var + " = fma(aa_" + std::to_string(m) + ", bb_" + std::to_string(n) + ", " + sum_var + ");";
        }
    }

    res += "ptrB += N; }";
    res += "ptrA += SUBGROUP_SIZE; }";

    return res;
}

std::string LoRAKernelOpt::GenerateStoreResultCode(size_t M, size_t N, Datatype dtype, bool is_A_kernel) const {
    std::string res = "";

    if (is_A_kernel) {
        for (size_t n = 0; n < N; ++n) {
            res += "ACCUMULATOR_TYPE alpha_" + std::to_string(n) + " = "
                + "TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(" + GenerateBlockRead(dtype, "alpha_ptr + " + std::to_string(n) + " * SUBGROUP_SIZE") + "));";
        }

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                res += GenerateBlockWrite(dtype,
                                          "ptrC + SUBGROUP_SIZE * " + std::to_string(n),
                                          "sum_" + std::to_string(m) + "_" + std::to_string(n) + " * alpha_" + std::to_string(n));
            }
            res += "ptrC += N;";
        }
    } else {
        for (size_t n = 0; n < N; ++n) {
            res += "INPUT0_TYPE main_N_" + std::to_string(n) + " = 0;";
        }

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                res += "main_N_" + std::to_string(n) + " = "
                    + "AS_INPUT0_TYPE(" + GenerateBlockRead(dtype, "main_ptr + " + std::to_string(n) + " * SUBGROUP_SIZE") + ");";

                res += GenerateBlockWrite(dtype,
                                          "ptrC + " + std::to_string(n) + " * SUBGROUP_SIZE",
                                          "TO_INPUT0_TYPE(sum_" + std::to_string(m) + "_" + std::to_string(n) + ") + main_N_" + std::to_string(n));
            }
            res += "main_ptr += N;";
            res += "ptrC += N;";
        }
    }

    return res;
}

JitConstants LoRAKernelOpt::GetJitConstants(const lora_params& params, size_t kernel_idx) const {
    auto jit = Parent::GetJitConstants(params);

    const auto& tuning_params = GetTuningParams(params, kernel_idx);
    size_t max_workgroup_size = params.engineInfo.maxWorkGroupSize;
    auto in_dtype = params.inputs[0].GetDType();

    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", tuning_params.subgroup_size));
    jit.AddConstant(MakeJitConstant("MAX_WORKGROUP_SIZE", max_workgroup_size));
    jit.AddConstant(MakeJitConstant("MAX_LORA_RANK", 256));

    if (kernel_idx <= KernelsTypes::FIRST_TOKEN_A_LARGE) {
        jit.AddConstant(MakeJitConstant("FIRST_TOKEN_A", 1));
    } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_B_MEDIUM ||
               kernel_idx == KernelsTypes::FIRST_TOKEN_B_LARGE) {
        jit.AddConstant(MakeJitConstant("FIRST_TOKEN_B", 1));
    } else if (kernel_idx == KernelsTypes::SECOND_TOKEN_A) {
        jit.AddConstant(MakeJitConstant("SECOND_TOKEN_A", 1));
    } else if (kernel_idx == KernelsTypes::SECOND_TOKEN_B) {
        jit.AddConstant(MakeJitConstant("SECOND_TOKEN_B", 1));
    } else if (kernel_idx == KernelsTypes::FUSED_OPS) {
        jit.AddConstant(MakeJitConstant("FUSED_OPS_KERNEL", 1));
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

        jit.AddConstant(MakeJitConstant("MAX_GEMMA_SGK", tuning_params.max_gemma_sgk));

        jit.AddConstant(MakeJitConstant("MAX_GEMMA_SG_BK", 64));
    } else if (kernel_idx <= KernelsTypes::FIRST_TOKEN_A_LARGE) {
        jit.AddConstant(MakeJitConstant("REG_M", tuning_params.regM));
        jit.AddConstant(MakeJitConstant("REG_N", tuning_params.regN));

        DimensionAccessHelperJit state_a_dims(params.inputs[2]);
        jit.AddConstant(MakeJitConstant("K", state_a_dims.f()));

        DimensionAccessHelperJit lora_input(params.inputs[1]);
        jit.AddConstant(MakeJitConstant("M", "(" + lora_input.b() + " * " + lora_input.f() + ")"));

        jit.AddConstant(MakeJitConstant("N", "LORA_RANK"));

        jit.AddConstant(MakeJitConstant("MAIN_MATMUL_CODE", GenerateMatMulCode(tuning_params.regM, tuning_params.regN, in_dtype, true)));

        jit.AddConstant(MakeJitConstant("MULTIPLY_AND_STORE_CODE", GenerateStoreResultCode(tuning_params.regM, tuning_params.regN, in_dtype, true)));
    } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_B_MEDIUM || kernel_idx == KernelsTypes::FIRST_TOKEN_B_LARGE) {
        jit.AddConstant(MakeJitConstant("REG_M", tuning_params.regM));
        jit.AddConstant(MakeJitConstant("REG_N", tuning_params.regN));

        DimensionAccessHelperJit lora_input(params.inputs[1]);
        jit.AddConstant(MakeJitConstant("M", "(" + lora_input.b() + " * " + lora_input.f() + ")"));

        DimensionAccessHelperJit state_b_dims(params.inputs[4]);
        jit.AddConstant(MakeJitConstant("N", state_b_dims.b()));

        jit.AddConstant(MakeJitConstant("K", "LORA_RANK"));

        jit.AddConstant(MakeJitConstant("MAIN_MATMUL_CODE", GenerateMatMulCode(tuning_params.regM, tuning_params.regN, in_dtype, false)));

        jit.AddConstant(MakeJitConstant("ADD_AND_STORE_CODE", GenerateStoreResultCode(tuning_params.regM, tuning_params.regN, in_dtype, false)));
    } else if (kernel_idx == KernelsTypes::FUSED_OPS) {
        if (!params.fused_ops.empty()) {
            FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "output[output_idx]", params.outputs[0].GetDType()};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }

    return jit;
}

void LoRAKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const lora_params&>(params);
        OPENVINO_ASSERT(kd.kernels.size() == KernelsTypes::TOTAL_KERNELS_NUM, "[GPU] Invalid kernels size for update dispatch data func");

        auto set_kernel_data = [&](size_t kernel_idx, bool skip_execution) {
            auto dispatch_data = SetDefault(prim_params, kernel_idx);
            kd.kernels[kernel_idx].params.workGroups.global = dispatch_data.gws;
            kd.kernels[kernel_idx].params.workGroups.local = dispatch_data.lws;
            kd.kernels[kernel_idx].skip_execution = skip_execution;
        };

        const auto& execute_kernels = GetSuitableKernels(prim_params);
        for (size_t kernel_idx = 0; kernel_idx < KernelsTypes::TOTAL_KERNELS_NUM; ++kernel_idx) {
            bool skip_execution = KernelData::SkipKernelExecution(prim_params);
            skip_execution |= kernel_idx != execute_kernels.first && kernel_idx != execute_kernels.second;

            if (kernel_idx == KernelsTypes::FUSED_OPS) {
                skip_execution = prim_params.fused_ops.empty();
            }
            set_kernel_data(kernel_idx, skip_execution);
        }

        kd.internalBuffers.clear();

        size_t input_state = prim_params.inputs[2].Feature().v;
        size_t lora_rank = prim_params.inputs[3].Feature().v;
        size_t max_workgroup_size = params.engineInfo.maxWorkGroupSize;
        size_t output_a_size = CeilDiv(input_state, gemm_a_sg_bk * (max_workgroup_size / std::max(lora_rank, static_cast<size_t>(1))));

        kd.internalBuffers.push_back(output_a_size * prim_params.inputs[0].ElementSize());
        kd.internalBufferDataType = GetAccumulatorType(prim_params);

        const auto& lora_input = prim_params.inputs[1];
        size_t batch = lora_input.Batch().v * lora_input.Feature().v;
        kd.internalBuffers.push_back(lora_rank * batch * prim_params.inputs[0].ElementSize());
    };
}

KernelsData LoRAKernelOpt::GetKernelsData(const Params& params) const {
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
            kd.internalBufferDataType = GetAccumulatorType(prim_params);
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
        } else if (kernel_idx <= KernelsTypes::FIRST_TOKEN_A_LARGE) {
            FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                             "", false, false, 3, 0, 0, prim_params.is_shape_agnostic);
            args.clear();
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT, 1});
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});
            args.push_back({ArgumentDescriptor::Types::INPUT, 3});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

            if (kd.internalBuffers.size() < 2) {
                size_t default_bytes_count = 1;
                kd.internalBuffers.push_back(default_bytes_count);
            }
        } else if (kernel_idx == KernelsTypes::FIRST_TOKEN_B_MEDIUM ||
                   kernel_idx == KernelsTypes::FIRST_TOKEN_B_LARGE) {
            FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                             "", false, false, 2, 0, 1, prim_params.is_shape_agnostic);
            args.clear();
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT, 0});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        } else if (kernel_idx == KernelsTypes::FUSED_OPS) {
            FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                             "", false, false, 0, GetFusedPrimitiveInputsCount(params), 1, prim_params.is_shape_agnostic);
        }
    }

    return { kd };
}

KernelsPriority LoRAKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

}  // namespace kernel_selector
