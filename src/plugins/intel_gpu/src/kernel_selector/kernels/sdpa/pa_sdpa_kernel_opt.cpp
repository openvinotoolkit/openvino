// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_opt.h"
#include "pa_sdpa_kernel_opt.h"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {
enum KernelsTypes {
    SINGLE_TOKEN = 0,
    SINGLE_TOKEN_GQA,
    MULTI_TOKENS,
    FINALIZATION,
    FINALIZATION_MULTI_TOKENS,
    SCORES_CALCULATION,
    TOTAL_KERNELS_NUM
};

static size_t get_heads_per_wi(const pa_sdpa_params& params) {
    if (params.conf.kv_group_size > 1) {
        std::vector<size_t> preferable_head_nums = {4, 3, 2};
        for (const auto& heads_num : preferable_head_nums) {
            const auto leftovers = params.conf.kv_group_size % heads_num;
            if (leftovers == 0 || heads_num - leftovers <= 1) {
                return heads_num;
            }
        }
    }

    return 1;
}

constexpr size_t subgroup_size = 16;
constexpr size_t seq_len_partition_size = 256;
constexpr size_t paged_attention_block_size = 16;
constexpr Datatype softmax_acc_dt = Datatype::F32;

size_t get_sg_number_scale_factor(const pa_sdpa_params& params, size_t head_size, size_t kernel_type) {
    if (params.conf.is_kv_compressed) {
        const size_t optimal_scale_factor = 2;
        if (kernel_type == KernelsTypes::SINGLE_TOKEN ||
            kernel_type == KernelsTypes::SINGLE_TOKEN_GQA ||
            kernel_type == KernelsTypes::MULTI_TOKENS) {
            if (head_size * optimal_scale_factor <= params.engineInfo.maxWorkGroupSize) {
                return optimal_scale_factor;
            }
        }
    }

    return 1;
}
}  // namespace

static std::string GetKernelName(std::string base_name, KernelsTypes type) {
    auto kernel_name = base_name;

    if (type == KernelsTypes::SINGLE_TOKEN) {
        kernel_name += "_single_token";
    } else if (type == KernelsTypes::SINGLE_TOKEN_GQA) {
        kernel_name += "_single_token_gqa";
    } else if (type == KernelsTypes::MULTI_TOKENS) {
        kernel_name += "_multi_tokens_seq";
    } else if (type == KernelsTypes::FINALIZATION) {
        kernel_name += "_finalization";
    } else if (type == KernelsTypes::FINALIZATION_MULTI_TOKENS) {
        kernel_name += "_finalization_multi_tokens_seq";
    } else if (type == KernelsTypes::SCORES_CALCULATION) {
        kernel_name += "_scores_calculation";
    }

    return kernel_name;
}

KernelsData PagedAttentionSDPAKernelOpt::GetKernelsData(const Params& p) const {
    if (!Validate(p)) {
        return {};
    }

    const auto& params = static_cast<const pa_sdpa_params&>(p);
    std::vector<KernelsTypes> kernels_type = { KernelsTypes::SINGLE_TOKEN,
                                               KernelsTypes::SINGLE_TOKEN_GQA,
                                               KernelsTypes::MULTI_TOKENS,
                                               KernelsTypes::FINALIZATION,
                                               KernelsTypes::FINALIZATION_MULTI_TOKENS };

    const auto has_scores_output = params.outputs.size() > 1;
    if (has_scores_output) {
        kernels_type.push_back(KernelsTypes::SCORES_CALCULATION);
    }

    KernelData kd = KernelData::Default<pa_sdpa_params>(params, kernels_type.size());
    kd.needs_sub_kernels_sync = true;

    GetUpdateDispatchDataFunc(kd);

    size_t kd_kernels_idx = 0;
    for (const auto& kernel_type : kernels_type) {
        const auto dispatch_data = SetDefault(params);
        const auto kernel_name = GetKernelName(kernelName, static_cast<KernelsTypes>(kernel_type));
        const auto entry_point = GetEntryPoint(kernel_name, params.layerID, params);
        auto jit_constants = GetJitConstants(params, kernel_type);

        const auto jit = CreateJit(kernel_name, jit_constants, entry_point);

        int inputs_num = static_cast<int>(params.inputs.size());
        int outputs_num = 1;
        if (kernel_type == KernelsTypes::SINGLE_TOKEN || kernel_type == KernelsTypes::SINGLE_TOKEN_GQA) {
            // SINGLE_TOKEN kernel doesn't use the subsequence_begins input
            inputs_num -= 1;
        } else if (kernel_type == KernelsTypes::FINALIZATION) {
            // FINALIZATION kernel uses only the past_lens data input
            inputs_num = 1;
        } else if (kernel_type == KernelsTypes::FINALIZATION_MULTI_TOKENS) {
            // FINALIZATION_MULTI_TOKENS kernel uses past_lens data input and subsequence_begins
            inputs_num = 2;
        } else if (kernel_type == KernelsTypes::SCORES_CALCULATION) {
            // SCORES_CALCULATION kernel uses past_lens data input and subsequence_begins
            inputs_num = 2;
            // Output is configured manually to use the second output memory buffer
            outputs_num = 0;
        }

        auto& kernel = kd.kernels[kd_kernels_idx++];
        FillCLKernelData(kernel,
                         dispatch_data,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entry_point,
                         {},
                         false,
                         false,
                         inputs_num,
                         GetFusedPrimitiveInputsCount(params),
                         outputs_num,
                         params.is_shape_agnostic);

        if (kernel_type == KernelsTypes::SCORES_CALCULATION) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
        }

        uint32_t internal_buffers_num = 0;
        if (has_scores_output) {
            // Intermediate softmax results for scores output calculation and precalculated accumulated
            // sequence length offsets for each subsequence
            internal_buffers_num += 2;
        }

        // Softmax's exp_sums, max_logits and intermediate output
        internal_buffers_num += 3;

        if (kernel_type == KernelsTypes::MULTI_TOKENS || kernel_type == KernelsTypes::FINALIZATION_MULTI_TOKENS) {
            // MULTIPLE_TOKENS kernels needs additional information related to mapping
            // launched kernel instances to subsequence indexes
            internal_buffers_num++;
        }

        for (uint32_t i = 0; i < internal_buffers_num; i++) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, i});
        }

        if (kernel_type == KernelsTypes::FINALIZATION || kernel_type == KernelsTypes::FINALIZATION_MULTI_TOKENS) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

            // Remove unused shape_info argument at finalization stage
            kernel.params.arguments.erase(kernel.params.arguments.begin());
        }

        if (kernel_type == KernelsTypes::SCORES_CALCULATION) {
            // The scores kernel needs to know if the current execution mode is mixed or ordinary
            // to configure proper memory access
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

            // Remove unused shape_info argument for scores kernel
            kernel.params.arguments.erase(kernel.params.arguments.begin());
        }
    }

    return {kd};
}

ParamsKey PagedAttentionSDPAKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

bool PagedAttentionSDPAKernelOpt::Validate(const Params& p) const {
    if (p.GetType() != KernelType::PA_SDPA)
        return false;

    const auto& params = static_cast<const pa_sdpa_params&>(p);
    if (!params.conf.is_paged_attention)
        return false;

    if (seq_len_partition_size % params.conf.paged_attention_block_size != 0)
        return false;

    if (params.conf.head_size % subgroup_size != 0)
        return false;

    const auto subgroups_per_wg = params.conf.head_size / subgroup_size;
    if (subgroups_per_wg > subgroup_size)
        return false;

    return true;
}

JitConstants PagedAttentionSDPAKernelOpt::GetJitConstants(const pa_sdpa_params& params, size_t kernel_idx) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& config = params.conf;
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", config.head_size));
    jit.AddConstant(MakeJitConstant("HEADS_NUM", config.heads_num));
    jit.AddConstant(MakeJitConstant("KV_HEADS_NUM", config.kv_heads_num));
    jit.AddConstant(MakeJitConstant("KV_HEADS_GROUP_SIZE", config.kv_group_size));
    jit.AddConstant(MakeJitConstant("SEQ_LEN_PARTITION_SIZE", seq_len_partition_size));
    jit.AddConstant(MakeJitConstant("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size));
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("SLIDING_WINDOW_SIZE", config.paged_attention_sliding_window));
    jit.AddConstant(MakeJitConstant("IS_KV_COMPRESSED", params.conf.is_kv_compressed));
    jit.AddConstant(MakeJitConstant("SG_SCALE_FACTOR", get_sg_number_scale_factor(params, config.head_size, kernel_idx)));
    jit.AddConstant(MakeJitConstant("XE2_QK_MULTIPLICATION", params.engineInfo.arch == gpu_arch::xe2));

    if (params.conf.is_kv_compressed) {
        auto scales_zp_size = params.inputs[0].ElementSize() * 2; // scale + zp
        jit.AddConstant(MakeJitConstant("SCALE_ZP_SIZE_PER_TOKEN", scales_zp_size));
        jit.AddConstant(MakeJitConstant("ADJUSTED_HEAD_SIZE", params.conf.head_size + scales_zp_size));
    } else {
        jit.AddConstant(MakeJitConstant("ADJUSTED_HEAD_SIZE", params.conf.head_size));
    }

    if (kernel_idx == KernelsTypes::SINGLE_TOKEN_GQA) {
        auto heads_per_wi = get_heads_per_wi(params);
        jit.AddConstant(MakeJitConstant("HEADS_PER_WI", heads_per_wi));
        jit.AddConstant(MakeJitConstant("ITERATIONS_PER_KV_HEADS_GROUP", CeilDiv(config.kv_group_size, heads_per_wi)));
        jit.AddConstant(MakeJitConstant("HEADS_LEFTOVERS_NUM", config.kv_group_size % heads_per_wi));
    } else {
        jit.AddConstant(MakeJitConstant("HEADS_PER_WI", 1));
    }

    auto sdpa_stage = 0;
    if (kernel_idx == KernelsTypes::FINALIZATION || kernel_idx == KernelsTypes::FINALIZATION_MULTI_TOKENS) {
        sdpa_stage = 1;
    } else if (kernel_idx == KernelsTypes::SCORES_CALCULATION) {
        sdpa_stage = 2;
    }
    jit.AddConstant(MakeJitConstant("SDPA_STAGE_" + std::to_string(sdpa_stage), 1));

    if (config.has_const_scale_val) {
        jit.AddConstant(MakeJitConstant("SCALE_VAL", config.scale_val));
    } else {
        const size_t scale_input_idx = 7;
        jit.AddConstant(MakeJitConstant("HAS_SCALE_INPUT", 1));
        jit.Merge(MakeTypeJitConstants(params.inputs[scale_input_idx].GetDType(), "SCALE_INPUT"));
    }

    if (params.conf.has_alibi_input) {
        const size_t alibi_input_idx = config.has_const_scale_val ? 7 : 8;
        jit.AddConstant(MakeJitConstant("HAS_ALIBI", 1));
        jit.Merge(MakeTypeJitConstants(params.inputs[alibi_input_idx].GetDType(), "ALIBI_INPUT"));
    }

    if (params.outputs.size() > 1) {
        jit.AddConstant(MakeJitConstant("PAGED_ATTENTION_SCORES_OUTPUT", 1));
    }

    if (params.conf.has_rotated_blocks)
        jit.AddConstant(MakeJitConstant("HAS_ROTATED_BLOCKS", 1));

    if (kernel_idx == KernelsTypes::MULTI_TOKENS || kernel_idx == KernelsTypes::FINALIZATION_MULTI_TOKENS)
        jit.AddConstant(MakeJitConstant("MULTI_TOKENS_PROCESSING", 1));

    jit.Merge(MakeTypeJitConstants(softmax_acc_dt, "SOFTMAX_ACCUMULATOR"));

    return jit;
}

CommonDispatchData PagedAttentionSDPAKernelOpt::SetDefault(const pa_sdpa_params& params, size_t kernel_idx) {
    CommonDispatchData dispatch_data;

    const auto& input = params.inputs[0];
    if (!input.is_dynamic()) {
        const size_t total_tokens = input.Batch().v;
        const size_t num_of_partitions = CeilDiv(params.conf.paged_attention_max_len, seq_len_partition_size);
        const size_t heads_num = static_cast<size_t>(params.conf.heads_num);
        const size_t head_size = static_cast<size_t>(params.conf.head_size);

        if (kernel_idx == KernelsTypes::SINGLE_TOKEN || kernel_idx == KernelsTypes::MULTI_TOKENS) {
            auto sg_scale = get_sg_number_scale_factor(params, head_size, kernel_idx);
            dispatch_data.gws = { total_tokens,
                                  heads_num,
                                  head_size * num_of_partitions * sg_scale };
            dispatch_data.lws = { 1, 1, head_size * sg_scale };
        } else if (kernel_idx == KernelsTypes::SINGLE_TOKEN_GQA) {
            auto sg_scale = get_sg_number_scale_factor(params, head_size, kernel_idx);

            auto kv_groups = heads_num / params.conf.kv_group_size;
            auto gqa_heads_num = kv_groups * CeilDiv(params.conf.kv_group_size, get_heads_per_wi(params));

            dispatch_data.gws = { total_tokens,
                                  gqa_heads_num,
                                  head_size * num_of_partitions * sg_scale };
            dispatch_data.lws = { 1, 1, head_size * sg_scale };
        } else if (kernel_idx == KernelsTypes::SCORES_CALCULATION) {
            const auto& past_lens = params.inputs[3];
            const auto subsequences_number = past_lens.Batch().v;

            size_t partition_size = 0;
            size_t num_of_partitions = 0;
            if (params.stage == PagedAttentionStage::PREFILL) {
                partition_size = SDPAKernelOpt::get_seq_len_partition_size(params, params.conf.head_size, 1);
            } else {
                partition_size = seq_len_partition_size;
            }

            num_of_partitions = CeilDiv(params.conf.paged_attention_max_len, partition_size);

            dispatch_data.gws = { partition_size * num_of_partitions,
                                  1,
                                  subsequences_number };
            dispatch_data.lws = { partition_size, 1, 1 };
        } else {
            dispatch_data.gws = { total_tokens,
                                  heads_num,
                                  head_size };
            dispatch_data.lws = { 1, 1, subgroup_size };
        }
    }

    return dispatch_data;
}

void PagedAttentionSDPAKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const pa_sdpa_params&>(params);

        const auto has_scores_output = prim_params.outputs.size() > 1;
        const auto expected_kernels_num = has_scores_output ? KernelsTypes::TOTAL_KERNELS_NUM : KernelsTypes::TOTAL_KERNELS_NUM - 1;
        OPENVINO_ASSERT(kd.kernels.size() == static_cast<size_t>(expected_kernels_num),
                        "[GPU] Invalid kernels size for update dispatch data func of SDPA kernel");

        const auto scores_calc_only = prim_params.stage == PagedAttentionStage::PREFILL && has_scores_output;
        const auto multi_tokens_mode = prim_params.stage == PagedAttentionStage::MIXED;

        // Apply GQA optimization starting from a certain sequence length (4K tokens) value
        const auto min_gqa_sequence_len = 16 * seq_len_partition_size;
        // Apply GQA only if there is a single subsequence in the request,
        // as multiple subsequences might have significantly different lengths
        const auto max_subsequences_num = 1;
        const auto subsequences_num = prim_params.inputs[0].Batch().v;
        const auto can_use_gqa_kernel = prim_params.conf.paged_attention_max_len >= static_cast<int64_t>(min_gqa_sequence_len) &&
                                        subsequences_num <= max_subsequences_num &&
                                        prim_params.conf.kv_group_size > 1 &&
                                        !multi_tokens_mode &&
                                        !scores_calc_only;

        auto dispatch_data = SetDefault(prim_params, KernelsTypes::SINGLE_TOKEN_GQA);
        kd.kernels[KernelsTypes::SINGLE_TOKEN_GQA].params.workGroups.global = dispatch_data.gws;
        kd.kernels[KernelsTypes::SINGLE_TOKEN_GQA].params.workGroups.local = dispatch_data.lws;
        kd.kernels[KernelsTypes::SINGLE_TOKEN_GQA].skip_execution = multi_tokens_mode || scores_calc_only || !can_use_gqa_kernel;

        dispatch_data = SetDefault(prim_params, KernelsTypes::SINGLE_TOKEN);
        kd.kernels[KernelsTypes::SINGLE_TOKEN].params.workGroups.global = dispatch_data.gws;
        kd.kernels[KernelsTypes::SINGLE_TOKEN].params.workGroups.local = dispatch_data.lws;
        kd.kernels[KernelsTypes::SINGLE_TOKEN].skip_execution = multi_tokens_mode || scores_calc_only || can_use_gqa_kernel;

        kd.kernels[KernelsTypes::MULTI_TOKENS].params.workGroups.global = dispatch_data.gws;
        kd.kernels[KernelsTypes::MULTI_TOKENS].params.workGroups.local = dispatch_data.lws;
        kd.kernels[KernelsTypes::MULTI_TOKENS].skip_execution = !multi_tokens_mode || scores_calc_only;

        size_t partition_size = 0;
        if (prim_params.stage == PagedAttentionStage::PREFILL) {
            partition_size = SDPAKernelOpt::get_seq_len_partition_size(params, prim_params.conf.head_size, 1);
        } else {
            partition_size = seq_len_partition_size;
        }
        const size_t num_of_partitions = CeilDiv(prim_params.conf.paged_attention_max_len, partition_size);

        dispatch_data = SetDefault(prim_params, KernelsTypes::FINALIZATION);
        kd.kernels[KernelsTypes::FINALIZATION].params.workGroups.global = dispatch_data.gws;
        kd.kernels[KernelsTypes::FINALIZATION].params.workGroups.local = dispatch_data.lws;
        kd.kernels[KernelsTypes::FINALIZATION].skip_execution = num_of_partitions == 1 || multi_tokens_mode || scores_calc_only;

        kd.kernels[KernelsTypes::FINALIZATION_MULTI_TOKENS].params.workGroups.global = dispatch_data.gws;
        kd.kernels[KernelsTypes::FINALIZATION_MULTI_TOKENS].params.workGroups.local = dispatch_data.lws;
        kd.kernels[KernelsTypes::FINALIZATION_MULTI_TOKENS].skip_execution = num_of_partitions == 1 || !multi_tokens_mode || scores_calc_only;

        ScalarDescriptor num_of_partitions_scalar;
        num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
        num_of_partitions_scalar.v.u32 = static_cast<uint32_t>(num_of_partitions);
        kd.kernels[KernelsTypes::FINALIZATION].params.scalars.resize(1);
        kd.kernels[KernelsTypes::FINALIZATION].params.scalars[0] = num_of_partitions_scalar;
        kd.kernels[KernelsTypes::FINALIZATION_MULTI_TOKENS].params.scalars.resize(1);
        kd.kernels[KernelsTypes::FINALIZATION_MULTI_TOKENS].params.scalars[0] = num_of_partitions_scalar;

        if (has_scores_output) {
            dispatch_data = SetDefault(prim_params, KernelsTypes::SCORES_CALCULATION);
            kd.kernels[KernelsTypes::SCORES_CALCULATION].params.workGroups.global = dispatch_data.gws;
            kd.kernels[KernelsTypes::SCORES_CALCULATION].params.workGroups.local = dispatch_data.lws;
            kd.kernels[KernelsTypes::SCORES_CALCULATION].skip_execution = false;

            ScalarDescriptor is_mixed_mode;
            is_mixed_mode.t = ScalarDescriptor::Types::UINT32;
            is_mixed_mode.v.u32 = static_cast<uint32_t>(multi_tokens_mode);
            kd.kernels[KernelsTypes::SCORES_CALCULATION].params.scalars.resize(1);
            kd.kernels[KernelsTypes::SCORES_CALCULATION].params.scalars[0] = is_mixed_mode;
        }

        const auto& input = prim_params.inputs[0];
        const size_t total_tokens = input.Batch().v;

        auto buf_dt_size = BytesPerElement(softmax_acc_dt);
        auto buf_elements_count = total_tokens * prim_params.conf.heads_num * num_of_partitions;
        auto buf_size = buf_elements_count * buf_dt_size;

        auto tmp_out_dt_size = BytesPerElement(softmax_acc_dt);
        auto tmp_out_elements_count = total_tokens * prim_params.conf.heads_num * prim_params.conf.head_size * num_of_partitions;
        auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

        const bool lockable = true;
        kd.internalBuffers.clear();

        if (has_scores_output) {
            const auto& past_lens = prim_params.inputs[3];
            auto subsequences_number = past_lens.Batch().v;
            auto softmax_buf_dt_size = BytesPerElement(softmax_acc_dt);

            auto softmax_buf_elements_count = subsequences_number * prim_params.conf.heads_num * num_of_partitions * partition_size;
            auto softmax_buf_size = softmax_buf_elements_count * softmax_buf_dt_size;

            // Softmax intermediate output
            kd.internalBuffers.emplace_back(softmax_buf_size, !lockable);
            // Precalculated accumulated sequence length offsets for each subsequence
            kd.internalBuffers.emplace_back(subsequences_number * BytesPerElement(Datatype::INT32), lockable);

            if (prim_params.stage == PagedAttentionStage::PREFILL) {
                // Recalculate buf_size as in case of PREFILL stage it's not needed to allocate buffer per each input token
                buf_elements_count = subsequences_number * prim_params.conf.heads_num * num_of_partitions;
                buf_size = buf_elements_count * buf_dt_size;

                // Intermediate tmp output buffer is not used for PREFILL stage
                tmp_out_size = tmp_out_dt_size;
            }
        }

        kd.internalBuffers.emplace_back(buf_size, !lockable); // softmax exp_sums
        kd.internalBuffers.emplace_back(buf_size, !lockable); // softmax max_logits
        kd.internalBuffers.emplace_back(tmp_out_size, !lockable); // intermediate output
        kd.internalBufferDataType = softmax_acc_dt;

        if (multi_tokens_mode) {
            auto buf_dt_size = BytesPerElement(Datatype::INT32);
            auto buf_elements_count = total_tokens;
            auto buf_size = Align(buf_elements_count * buf_dt_size, BytesPerElement(softmax_acc_dt));
            kd.internalBuffers.emplace_back(buf_size, lockable);
        }
    };
}

}  // namespace kernel_selector
