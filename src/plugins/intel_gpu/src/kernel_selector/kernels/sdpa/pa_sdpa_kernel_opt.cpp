// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_sdpa_kernel_opt.h"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

// For kernel w/o split
constexpr size_t max_sequence_length = 3072;

constexpr size_t seq_len_partition_size = 256;
constexpr size_t subgroup_size = 16;

const Datatype softmax_acc_dt = Datatype::F32;

// Use flash attention or not
const bool use_seq_len_split = true;

void PagedAttentionSDPAKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const pa_sdpa_params&>(params);

        const size_t expected_kernels_num = use_seq_len_split ? 2 : 1;
        OPENVINO_ASSERT(kd.kernels.size() == expected_kernels_num, "[GPU] Invalid kernels size for update dispatch data func of SDPA kernel");

        auto dispatchData = SetDefault(prim_params, 0);
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;

        if (expected_kernels_num == 2) {
            const auto& input = prim_params.inputs[0];
            const size_t batch_size = input.Batch().v;
            // const size_t seq_len = input.Feature().v;
            const size_t sequences_number = batch_size;
            const size_t num_of_partitions = CeilDiv(prim_params.max_context_len, seq_len_partition_size);

            // const size_t num_of_partitions = 1;

            auto dispatchData = SetDefault(prim_params, 1);
            kd.kernels[1].params.workGroups.global = dispatchData.gws;
            kd.kernels[1].params.workGroups.local = dispatchData.lws;
            // Write directly to the output in SDPA main kernel in case of single portion
            kd.kernels[1].skip_execution = num_of_partitions == 1;

            auto buf_dt_size = 4;
            auto buf_elements_count = sequences_number * prim_params.conf.heads_num * num_of_partitions;
            // auto buf_elements_count = 1;
            auto buf_size = buf_elements_count * buf_dt_size;

            auto tmp_out_dt_size = 4;
            auto tmp_out_elements_count = sequences_number * prim_params.conf.heads_num * prim_params.conf.head_size * num_of_partitions;
            // auto tmp_out_elements_count = 1;
            auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

            kd.internalBufferSizes.clear();
            kd.internalBufferSizes.push_back(buf_size);
            kd.internalBufferSizes.push_back(buf_size);
            kd.internalBufferSizes.push_back(tmp_out_size);
            kd.internalBufferDataType = softmax_acc_dt;

            ScalarDescriptor num_of_partitions_scalar;
            num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
            num_of_partitions_scalar.v.u32 = num_of_partitions;

            kd.kernels[1].params.scalars.resize(1);
            kd.kernels[1].params.scalars[0] = num_of_partitions_scalar;
        }
    };
}

KernelsData PagedAttentionSDPAKernelOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const uint kernels_num = use_seq_len_split ? 2 : 1;
    KernelData kd = KernelData::Default<pa_sdpa_params>(params, kernels_num);
    kd.needs_sub_kernels_sync = true;
    GetUpdateDispatchDataFunc(kd);

    const auto& kernel_params = static_cast<const pa_sdpa_params&>(params);

    for (size_t i = 0; i < kernels_num; i++) {
        const auto dispatch_data = SetDefault(kernel_params);
        const auto kernel_name = i == 0 ? kernelName : "pa_sdpa_finalization";
        const auto entry_point = GetEntryPoint(kernel_name, kernel_params.layerID, params, i);
        auto jit_constants = GetJitConstants(kernel_params);

        jit_constants.AddConstant(MakeJitConstant(i == 0 ? "SDPA_STAGE_0" : "SDPA_STAGE_1", 1));

        const auto jit = CreateJit(kernel_name, jit_constants, entry_point);

        const size_t inputs_num = i == 0 ? static_cast<int>(kernel_params.inputs.size()) : 1;
        auto& kernel = kd.kernels[i];
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
                        GetFusedPrimitiveInputsCount(kernel_params),
                        static_cast<int>(kernel_params.outputs.size()),
                        kernel_params.is_shape_agnostic);

        if (use_seq_len_split) {
            if (i == 1) // Remove unused shape_info argument
                kernel.params.arguments.erase(kernel.params.arguments.begin());

            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 6});
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 7});
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 8});
            kd.internalBufferDataType = softmax_acc_dt;

            if (i == 1) {
                kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
            }
        }
    }

    return {kd};
}

ParamsKey PagedAttentionSDPAKernelOpt::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableInputDataType(Datatype::INT32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::INT32);
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableInputLayout(DataLayout::bfzyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfzyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDynamicShapesSupport();
    key.EnableDifferentTypes();
    return key;
}

bool PagedAttentionSDPAKernelOpt::Validate(const Params& params) const {
    if (params.GetType() != KernelType::PA_SDPA) {
        return false;
    }

    // const auto& kernel_params = dynamic_cast<const pa_sdpa_params&>(params);
    // if (seq_len_partition_size % kernel_params.configuration.block_size != 0)
    //     return false;

    // if (kernel_params.configuration.head_size % subgroup_size != 0)
    //     return false;

    // const auto subgroups_per_wg = kernel_params.configuration.head_size / subgroup_size;
    // if (subgroups_per_wg > subgroup_size)
    //     return false;

    return true;
}

JitConstants PagedAttentionSDPAKernelOpt::GetJitConstants(const pa_sdpa_params& kernel_params) const {
    JitConstants jit = MakeBaseParamsJitConstants(kernel_params);

    const auto& config = kernel_params.conf;
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", config.head_size));
    jit.AddConstant(MakeJitConstant("HEADS_NUM", config.heads_num));
    jit.AddConstant(MakeJitConstant("KV_HEADS_NUM", config.kv_heads_num));
    jit.AddConstant(MakeJitConstant("NUM_QUERIES_PER_KV_HEAD", config.heads_num / config.kv_heads_num));
    jit.AddConstant(MakeJitConstant("VLLM_BLOCK_SIZE", 16));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", 1));

    if (config.has_scale_val) {
        jit.AddConstant(MakeJitConstant("SCALE_VAL", config.scale_val));
    }

    jit.Merge(MakeTypeJitConstants(softmax_acc_dt, "SOFTMAX_ACCUMULATOR"));


    if (use_seq_len_split) {
        jit.AddConstant(MakeJitConstant("USE_SEQ_LEN_SPLIT", true));
        jit.AddConstant(MakeJitConstant("SEQ_LEN_PARTITION_SIZE", seq_len_partition_size));
        jit.AddConstant(MakeJitConstant("SHARED_MEM_SIZE", seq_len_partition_size));
    } else {
        jit.AddConstant(MakeJitConstant("SHARED_MEM_SIZE", max_sequence_length));
    }

    return jit;
}

CommonDispatchData PagedAttentionSDPAKernelOpt::SetDefault(const pa_sdpa_params& kernel_params, size_t kernel_idx) {
    CommonDispatchData dispatch_data;


    const auto& input = kernel_params.inputs[0];
    if (!input.is_dynamic()) {
        const size_t seq_num = input.Batch().v;
        // const size_t seq_len = input.Feature().v;
        // const size_t tokens_num = batch_size * seq_len;

        const size_t num_of_partitions =
            use_seq_len_split ? CeilDiv(kernel_params.max_context_len, seq_len_partition_size) : 1;

        // const size_t num_of_partitions = 1;
        const size_t heads_num = static_cast<size_t>(kernel_params.conf.heads_num);
        const size_t head_size = static_cast<size_t>(kernel_params.conf.head_size);

        if (kernel_idx == 0) {
            dispatch_data.gws = { seq_num,
                                  heads_num,
                                  head_size * num_of_partitions };
            dispatch_data.lws = { 1, 1, head_size };
        } else {
            dispatch_data.gws = { seq_num,
                                  heads_num,
                                  head_size };
            dispatch_data.lws = { 1, 1, subgroup_size };
        }
    }

    return dispatch_data;
}

}  // namespace kernel_selector
