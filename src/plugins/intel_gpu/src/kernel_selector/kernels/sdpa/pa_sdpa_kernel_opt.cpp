// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_sdpa_kernel_opt.h"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {
enum KernelsTypes {
    SINGLE_TOKEN = 0,
    FINALIZATION,
    TOTAL_KERNELS_NUM
};

constexpr size_t subgroup_size = 16;
constexpr size_t seq_len_partition_size = 256;
constexpr size_t paged_attention_block_size = 16;
constexpr Datatype softmax_acc_dt = Datatype::F32;
}  // namespace

static std::string GetKernelName(std::string base_name, KernelsTypes type) {
    auto kernel_name = base_name;

    if (type == KernelsTypes::SINGLE_TOKEN) {
        kernel_name += "_single_token";
    } else if (type == KernelsTypes::FINALIZATION) {
        kernel_name += "_finalization";
    }

    return kernel_name;
}

KernelsData PagedAttentionSDPAKernelOpt::GetKernelsData(const Params& p) const {
    if (!Validate(p)) {
        return {};
    }

    const auto& params = static_cast<const pa_sdpa_params&>(p);
    const std::vector<KernelsTypes> kernels_type = { KernelsTypes::SINGLE_TOKEN,
                                                     KernelsTypes::FINALIZATION };

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

        const size_t inputs_num = kernel_type == KernelsTypes::SINGLE_TOKEN ? static_cast<int>(params.inputs.size()) : 1;
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
                         static_cast<int>(inputs_num),
                         GetFusedPrimitiveInputsCount(params),
                         static_cast<int>(params.outputs.size()),
                         params.is_shape_agnostic);

        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        kd.internalBufferDataType = softmax_acc_dt;

        if (kernel_type == KernelsTypes::FINALIZATION) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

            // Remove unused shape_info argument at finalization stage
            kernel.params.arguments.erase(kernel.params.arguments.begin());
        }
    }

    return {kd};
}

ParamsKey PagedAttentionSDPAKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
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
    jit.AddConstant(MakeJitConstant("SEQ_LEN_PARTITION_SIZE", seq_len_partition_size));
    jit.AddConstant(MakeJitConstant("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size));
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));

    if (config.broadcast_axis != -1) {
        jit.AddConstant(MakeJitConstant("BROADCAST_GROUP_SIZE", config.group_size));
    }

    auto sdpa_stage = kernel_idx == KernelsTypes::FINALIZATION ? 1 : 0;
    jit.AddConstant(MakeJitConstant("SDPA_STAGE_" + std::to_string(sdpa_stage), 1));

    if (config.has_scale_val)
        jit.AddConstant(MakeJitConstant("SCALE_VAL", config.scale_val));

    if (params.conf.has_alibi_input)
        jit.AddConstant(MakeJitConstant("HAS_ALIBI", 1));

    jit.Merge(MakeTypeJitConstants(softmax_acc_dt, "SOFTMAX_ACCUMULATOR"));

    return jit;
}

CommonDispatchData PagedAttentionSDPAKernelOpt::SetDefault(const pa_sdpa_params& params, size_t kernel_idx) {
    CommonDispatchData dispatch_data;

    const auto& input = params.inputs[0];
    if (!input.is_dynamic()) {
        const size_t sequences_number = input.Batch().v;
        const size_t num_of_partitions = CeilDiv(params.max_context_len, seq_len_partition_size);
        const size_t heads_num = static_cast<size_t>(params.conf.heads_num);
        const size_t head_size = static_cast<size_t>(params.conf.head_size);

        if (kernel_idx == 0) {
            dispatch_data.gws = { sequences_number,
                                  heads_num,
                                  head_size * num_of_partitions };
            dispatch_data.lws = { 1, 1, head_size };
        } else {
            dispatch_data.gws = { sequences_number,
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

        const size_t expected_kernels_num = 2;
        OPENVINO_ASSERT(kd.kernels.size() == expected_kernels_num, "[GPU] Invalid kernels size for update dispatch data func of SDPA kernel");

        auto dispatch_data1 = SetDefault(prim_params, KernelsTypes::SINGLE_TOKEN);
        kd.kernels[KernelsTypes::SINGLE_TOKEN].params.workGroups.global = dispatch_data1.gws;
        kd.kernels[KernelsTypes::SINGLE_TOKEN].params.workGroups.local = dispatch_data1.lws;
        kd.kernels[KernelsTypes::SINGLE_TOKEN].skip_execution = false;

        const auto& input = prim_params.inputs[0];
        const size_t sequences_number = input.Batch().v;
        const size_t num_of_partitions = CeilDiv(prim_params.max_context_len, seq_len_partition_size);

        auto dispatch_data2 = SetDefault(prim_params, KernelsTypes::FINALIZATION);
        kd.kernels[KernelsTypes::FINALIZATION].params.workGroups.global = dispatch_data2.gws;
        kd.kernels[KernelsTypes::FINALIZATION].params.workGroups.local = dispatch_data2.lws;
        kd.kernels[KernelsTypes::FINALIZATION].skip_execution = num_of_partitions == 1;

        ScalarDescriptor num_of_partitions_scalar;
        num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
        num_of_partitions_scalar.v.u32 = static_cast<uint32_t>(num_of_partitions);
        kd.kernels[KernelsTypes::FINALIZATION].params.scalars.resize(1);
        kd.kernels[KernelsTypes::FINALIZATION].params.scalars[0] = num_of_partitions_scalar;

        auto buf_dt_size = BytesPerElement(softmax_acc_dt);
        auto buf_elements_count = sequences_number * prim_params.conf.heads_num * num_of_partitions;
        auto buf_size = buf_elements_count * buf_dt_size;

        auto tmp_out_dt_size = BytesPerElement(softmax_acc_dt);
        auto tmp_out_elements_count = sequences_number * prim_params.conf.heads_num * prim_params.conf.head_size * num_of_partitions;
        auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(buf_size);
        kd.internalBufferSizes.push_back(buf_size);
        kd.internalBufferSizes.push_back(tmp_out_size);
        kd.internalBufferDataType = softmax_acc_dt;
    };
}

}  // namespace kernel_selector
