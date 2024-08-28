// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

namespace {
enum KernelsTypes {
    SINGLE_TOKEN = 0,
    MULTI_TOKENS,
    FINALIZATION,
    TOTAL_KERNELS_NUM
};

constexpr size_t subgroup_size = 16;
}  // namespace

static size_t get_sg_number_scale_factor(const sdpa_params& sdpa_params, size_t kernel_type) {
    if (kernel_type == KernelsTypes::MULTI_TOKENS) {
        const size_t optimal_scale_factor = 2;
        if (sdpa_params.conf.head_size * optimal_scale_factor <= sdpa_params.engineInfo.maxWorkGroupSize) {
            return optimal_scale_factor;
        }
    }

    return 1;
}

static size_t get_target_seq_len_block_size() {
    const size_t block_size = 16;
    return block_size;
}

static size_t get_seq_len_partition_size(const sdpa_params& sdpa_params, size_t kernel_type) {
    size_t seq_len = 0;
    if (kernel_type == KernelsTypes::MULTI_TOKENS) {
        seq_len = sdpa_params.conf.head_size * get_sg_number_scale_factor(sdpa_params, kernel_type);
    } else {
        seq_len = 256;
    }

    return seq_len;
}

static Datatype get_softmax_acc_type() {
    return Datatype::F32;
}

static bool is_prefill_stage(const sdpa_params& sdpa_params) {
    TransposedDimensionAccessHelperBase dims_q(sdpa_params.inputs[0], sdpa_params.input0_order);
    auto target_seq_len = dims_q.y_dim().v;

    return target_seq_len > 1;
}

static size_t get_partitions_num(const sdpa_params& sdpa_params, size_t kernel_type) {
    if (sdpa_params.has_dynamic_tensors() || kernel_type == KernelsTypes::MULTI_TOKENS)
        return 1;

    TransposedDimensionAccessHelperBase dims_k(sdpa_params.inputs[1], sdpa_params.input1_order);
    auto source_seq_len = dims_k.y_dim().v;

    return CeilDiv(source_seq_len, get_seq_len_partition_size(sdpa_params, kernel_type));
}

static std::vector<size_t> get_internal_buffer_sizes(const sdpa_params& sdpa_params, size_t kernel_type) {
    if (sdpa_params.conf.is_paged_attention) {
        const auto blocks_indexes_dt = Datatype::INT32;
        const auto target_seq_len_block_size = get_target_seq_len_block_size();
        const auto target_seq_len = sdpa_params.conf.paged_attention_aligned_seq_len;
        const auto blocks_indexes_buf_size = CeilDiv(target_seq_len, target_seq_len_block_size) * BytesPerElement(blocks_indexes_dt);

        return {blocks_indexes_buf_size};
    } else {
        if (sdpa_params.has_dynamic_tensors() || kernel_type == KernelsTypes::MULTI_TOKENS) {
            return {1, 1};
        } else {
            TransposedDimensionAccessHelperBase dims_q(sdpa_params.inputs[0], sdpa_params.input0_order);
            const auto& output = sdpa_params.outputs[0];

            const auto head_size = dims_q.x_dim().v;
            const auto num_of_partitions = get_partitions_num(sdpa_params, kernel_type);
            const auto is_prefill = is_prefill_stage(sdpa_params);

            const auto buf_dt_size = BytesPerElement(get_softmax_acc_type());
            const auto buf_elements_count = (num_of_partitions == 1 || is_prefill) ? 1 : output.LogicalSize() / head_size * num_of_partitions;
            const auto buf_size = buf_elements_count * buf_dt_size;

            const auto tmp_out_dt_size = output.ElementSize();
            const auto tmp_out_elements_count = (num_of_partitions == 1 || is_prefill) ? 1 : output.LogicalSize() * num_of_partitions;
            const auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

            return {buf_size, tmp_out_size};
        }
    }
}

static std::string GetKernelName(std::string base_name, KernelsTypes type, const sdpa_params& params) {
    const bool is_indirect = params.indirect_axis != -1;
    const bool is_paged_attention = params.conf.is_paged_attention;

    std::string kernel_name = "";
    if (is_paged_attention)
        kernel_name += "pa_";

    kernel_name += base_name;
    if (is_indirect)
        kernel_name += "_ind";

    if (type == KernelsTypes::SINGLE_TOKEN) {
        kernel_name += "_single_token";
    } else if (type == KernelsTypes::MULTI_TOKENS) {
        kernel_name += "_multi_tokens";
    } else if (type == KernelsTypes::FINALIZATION) {
        kernel_name += "_finalization";
    }

    return kernel_name;
}

ParamsKey SDPAKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);

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

bool SDPAKernelOpt::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        return false;

    const sdpa_params& params = static_cast<const sdpa_params&>(p);

    if (params.conf.head_size < 1 || params.conf.head_size % subgroup_size != 0)
        return false;

    return true;
}

JitConstants SDPAKernelOpt::GetJitConstants(const sdpa_params& params, size_t kernel_idx) const {
    auto jit = SDPAKernelBase::GetJitConstants(params);

    const auto softmax_acc_dt = get_softmax_acc_type();
    jit.Merge(MakeTypeJitConstants(softmax_acc_dt, "SOFTMAX_ACCUMULATOR"));

    const auto& config = params.conf;
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", config.head_size));
    jit.AddConstant(MakeJitConstant("SEQ_LEN_PARTITION_SIZE", get_seq_len_partition_size(params, kernel_idx)));

    auto target_seq_len_block_size = kernel_idx == KernelsTypes::SINGLE_TOKEN ? 1 : get_target_seq_len_block_size();
    jit.AddConstant(MakeJitConstant("TARGET_SEQ_LEN_BLOCK_SIZE", target_seq_len_block_size));

    auto sdpa_stage = kernel_idx == KernelsTypes::FINALIZATION ? 1 : 0;
    jit.AddConstant(MakeJitConstant("SDPA_STAGE_" + std::to_string(sdpa_stage), 1));
    jit.AddConstant(MakeJitConstant("SG_SCALE_FACTOR", get_sg_number_scale_factor(params, kernel_idx)));

    if (params.conf.is_paged_attention) {
        if (params.conf.has_alibi_input) {
            jit.AddConstant(MakeJitConstant("HAS_ALIBI", 1));
        }

        if (params.conf.has_scale_val) {
            jit.AddConstant(MakeJitConstant("STATIC_SCALE_VALUE_INV", 1.0f / params.conf.scale_val));
            jit.AddConstant(MakeJitConstant("STATIC_SCALE_VALUE", params.conf.scale_val));
        }
    } else if (params.inputs.size() <= 4) {
        jit.AddConstant(MakeJitConstant("STATIC_SCALE_VALUE_INV", std::sqrt(static_cast<float>(params.conf.head_size))));
        jit.AddConstant(MakeJitConstant("STATIC_SCALE_VALUE", 1.0f / std::sqrt(static_cast<float>(params.conf.head_size))));
    }

    if (params.conf.is_paged_attention)
        jit.AddConstant(MakeJitConstant("IS_PAGED_ATTENTION", 1));

    if (params.engineInfo.supports_immad && params.conf.broadcast_axis == -1 && params.conf.head_size >= 128)
        jit.AddConstant(MakeJitConstant("LOAD_KEY_LEFTOVERS_IN_CALC_LOOP", 1));

    return jit;
}

CommonDispatchData SDPAKernelOpt::SetDefault(const sdpa_params& params, size_t kernel_idx) const {
    CommonDispatchData dispatch_data;

    const auto& query_input = params.inputs[0];
    if (!query_input.is_dynamic()) {
        if (params.conf.is_paged_attention) {
            OPENVINO_ASSERT(kernel_idx == KernelsTypes::MULTI_TOKENS);

            const size_t sg_num_scale = get_sg_number_scale_factor(params, kernel_idx);
            const size_t heads_num = static_cast<size_t>(params.conf.heads_num);
            const size_t target_seq_len_block_size = get_target_seq_len_block_size();
            const size_t target_seq_len = static_cast<size_t>(params.conf.paged_attention_aligned_seq_len);
            const size_t head_size = static_cast<size_t>(params.conf.head_size);

            dispatch_data.gws = { heads_num,
                                  CeilDiv(target_seq_len, target_seq_len_block_size),
                                  head_size * sg_num_scale };
            dispatch_data.lws = { 1, 1, head_size * sg_num_scale };

            return dispatch_data;
        }

        TransposedDimensionAccessHelperBase dims_q(params.inputs[0], params.input0_order);
        TransposedDimensionAccessHelperBase output(params.outputs[0], params.output_order);

        const size_t batch_size = output.b_dim().v;
        const size_t heads_num = output.f_dim().v;
        const size_t target_seq_len = dims_q.y_dim().v;
        const size_t head_size = static_cast<size_t>(params.conf.head_size);
        const size_t num_of_partitions = get_partitions_num(params, kernel_idx);
        const size_t target_seq_len_block_size = kernel_idx == 1 ? get_target_seq_len_block_size() : 1;

        if (kernel_idx == KernelsTypes::SINGLE_TOKEN) {
            dispatch_data.gws = { batch_size * heads_num,
                                  CeilDiv(target_seq_len, target_seq_len_block_size),
                                  head_size * num_of_partitions };
            dispatch_data.lws = { 1, 1, head_size };
        } else if (kernel_idx == KernelsTypes::MULTI_TOKENS) {
            const size_t sg_num_scale = get_sg_number_scale_factor(params, kernel_idx);
            dispatch_data.gws = { batch_size * heads_num,
                                  CeilDiv(target_seq_len, target_seq_len_block_size),
                                  head_size * sg_num_scale };
            dispatch_data.lws = { 1, 1, head_size * sg_num_scale };
        } else if (kernel_idx == KernelsTypes::FINALIZATION) {
            dispatch_data.gws = { batch_size * heads_num,
                                  target_seq_len,
                                  subgroup_size };
            dispatch_data.lws = { 1, 1, subgroup_size };
        }
    }

    return dispatch_data;
}

KernelsData SDPAKernelOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    std::vector<KernelsTypes> kernels_type;
    const auto& prim_params = static_cast<const sdpa_params&>(params);

    if (prim_params.conf.is_paged_attention) {
        kernels_type = { KernelsTypes::MULTI_TOKENS };
    } else if (params.is_shape_agnostic) {
        kernels_type = { KernelsTypes::SINGLE_TOKEN, KernelsTypes::MULTI_TOKENS, KernelsTypes::FINALIZATION };
    } else {
        TransposedDimensionAccessHelperBase dims_q(prim_params.inputs[0], prim_params.input0_order);
        const auto is_prefill = is_prefill_stage(prim_params);

        if (is_prefill) {
            kernels_type = { KernelsTypes::MULTI_TOKENS };
        } else {
            kernels_type = { KernelsTypes::SINGLE_TOKEN, KernelsTypes::FINALIZATION };
        }
    }

    KernelData kd = KernelData::Default<sdpa_params>(params, kernels_type.size());
    kd.needs_sub_kernels_sync = true;

    GetUpdateDispatchDataFunc(kd);

    size_t kd_kernels_idx = 0;
    for (const auto& kernel_idx : kernels_type) {
        auto dispatch_data = SetDefault(prim_params, kernel_idx);
        auto kernel_name = GetKernelName(kernelName, static_cast<KernelsTypes>(kernel_idx), prim_params);
        auto entry_point = GetEntryPoint(kernel_name, prim_params.layerID, params);
        auto jit_constants = GetJitConstants(prim_params, kernel_idx);
        auto jit = CreateJit(kernel_name, jit_constants, entry_point);

        auto& kernel = kd.kernels[kd_kernels_idx++];

        auto inputs_num =
            kernel_idx == KernelsTypes::FINALIZATION ? 0 : static_cast<int>(prim_params.inputs.size());

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
                         static_cast<int>(prim_params.outputs.size()),
                         prim_params.is_shape_agnostic);

        if (prim_params.indirect_axis != -1 && kernel_idx != KernelsTypes::FINALIZATION)
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(prim_params.inputs.size())});

        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});

        const auto buf_sizes = get_internal_buffer_sizes(prim_params, kernel_idx);
        if (!prim_params.conf.is_paged_attention) {
            kd.internalBufferSizes.clear();
            kd.internalBufferSizes.push_back(buf_sizes[0]);
            kd.internalBufferSizes.push_back(buf_sizes[0]);
            kd.internalBufferSizes.push_back(buf_sizes[1]);
            kd.internalBufferDataType = prim_params.inputs[0].GetDType();
        }

        if (kernel_idx == KernelsTypes::FINALIZATION) {
            const auto num_of_partitions = get_partitions_num(prim_params, kernel_idx);
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

            ScalarDescriptor num_of_partitions_scalar;
            num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
            num_of_partitions_scalar.v.u32 = static_cast<uint32_t>(num_of_partitions);

            kernel.params.scalars.clear();
            kernel.params.scalars.push_back(num_of_partitions_scalar);
        }
    }

    return { kd };
}

void SDPAKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kernel_data) {
        const auto& prim_params = static_cast<const sdpa_params&>(params);

        const size_t paged_attention_kernels_num = 1;
        const size_t expected_kernels_num = prim_params.conf.is_paged_attention ? paged_attention_kernels_num
                                                                                : KernelsTypes::TOTAL_KERNELS_NUM;
        OPENVINO_ASSERT(kernel_data.kernels.size() == expected_kernels_num,
                        "[GPU] Invalid kernels size for update dispatch data func of SDPA kernel");

        if (prim_params.conf.is_paged_attention) {
            auto dispatch_data = SetDefault(prim_params, KernelsTypes::MULTI_TOKENS);
            kernel_data.kernels[0].params.workGroups.global = dispatch_data.gws;
            kernel_data.kernels[0].params.workGroups.local = dispatch_data.lws;
            kernel_data.kernels[0].skip_execution = false;
        } else {
            const auto num_of_partitions = get_partitions_num(prim_params, KernelsTypes::SINGLE_TOKEN);
            const auto buf_sizes = get_internal_buffer_sizes(prim_params, KernelsTypes::SINGLE_TOKEN);
            const auto is_prefill = is_prefill_stage(prim_params);

            ScalarDescriptor num_of_partitions_scalar;
            num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
            num_of_partitions_scalar.v.u32 = static_cast<uint32_t>(num_of_partitions);

            auto dispatch_data1 = SetDefault(prim_params, KernelsTypes::SINGLE_TOKEN);
            kernel_data.kernels[KernelsTypes::SINGLE_TOKEN].params.workGroups.global = dispatch_data1.gws;
            kernel_data.kernels[KernelsTypes::SINGLE_TOKEN].params.workGroups.local = dispatch_data1.lws;
            kernel_data.kernels[KernelsTypes::SINGLE_TOKEN].skip_execution = is_prefill;

            auto dispatch_data2 = SetDefault(prim_params, KernelsTypes::MULTI_TOKENS);
            kernel_data.kernels[KernelsTypes::MULTI_TOKENS].params.workGroups.global = dispatch_data2.gws;
            kernel_data.kernels[KernelsTypes::MULTI_TOKENS].params.workGroups.local = dispatch_data2.lws;
            kernel_data.kernels[KernelsTypes::MULTI_TOKENS].skip_execution = !is_prefill;

            auto dispatch_data3 = SetDefault(prim_params, KernelsTypes::FINALIZATION);
            kernel_data.kernels[KernelsTypes::FINALIZATION].params.workGroups.global = dispatch_data3.gws;
            kernel_data.kernels[KernelsTypes::FINALIZATION].params.workGroups.local = dispatch_data3.lws;
            kernel_data.kernels[KernelsTypes::FINALIZATION].skip_execution = is_prefill || num_of_partitions == 1;

            kernel_data.kernels[KernelsTypes::FINALIZATION].params.scalars.clear();
            kernel_data.kernels[KernelsTypes::FINALIZATION].params.scalars.push_back(num_of_partitions_scalar);

            kernel_data.internalBufferSizes.clear();
            kernel_data.internalBufferSizes.push_back(buf_sizes[0]);
            kernel_data.internalBufferSizes.push_back(buf_sizes[0]);
            kernel_data.internalBufferSizes.push_back(buf_sizes[1]);
            kernel_data.internalBufferDataType = prim_params.inputs[0].GetDType();
        }
    };
}

KernelsPriority SDPAKernelOpt::GetKernelsPriority(const Params& params) const {
    return params.engineInfo.supports_immad ?  FORCE_PRIORITY_2 : FORCE_PRIORITY_1;
}
}  // namespace kernel_selector
