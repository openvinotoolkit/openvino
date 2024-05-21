// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

constexpr size_t subgroup_size = 16;

enum KernelsTypes {
    SINGLE_TOKEN = 0,
    MULTI_TOKENS,
    FINALIZATION,
    TOTAL_KERNELS_NUM
};

static size_t get_target_seq_len_block_size() {
    const size_t block_size = 16;
    return block_size;
}


static size_t get_seq_len_partition_size() {
    const size_t seq_len = 256;
    return seq_len;
}

ParamsKey SDPAKernelOpt::GetSupportedKey() const {
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

    const auto softmax_acc_dt = params.inputs[0].GetDType();
    jit.Merge(MakeTypeJitConstants(softmax_acc_dt, "SOFTMAX_ACCUMULATOR"));

    const auto& config = params.conf;
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", config.head_size));
    jit.AddConstant(MakeJitConstant("SEQ_LEN_PARTITION_SIZE", get_seq_len_partition_size()));

    auto target_seq_len_block_size = kernel_idx == KernelsTypes::SINGLE_TOKEN ? 1 : get_target_seq_len_block_size();
    jit.AddConstant(MakeJitConstant("TARGET_SEQ_LEN_BLOCK_SIZE", target_seq_len_block_size));

    auto sdpa_stage = kernel_idx == KernelsTypes::FINALIZATION ? 1 : 0;
    jit.AddConstant(MakeJitConstant("SDPA_STAGE_" + std::to_string(sdpa_stage), 1));

    return jit;
}

CommonDispatchData SDPAKernelOpt::SetDefault(const sdpa_params& params, size_t kernel_idx) const {
    CommonDispatchData dispatch_data;

    const auto& query_input = params.inputs[0];

    if (!query_input.is_dynamic()) {
        TransposedDimensionAccessHelperBase dims_q(params.inputs[0], params.input0_order);
        TransposedDimensionAccessHelperBase dims_k(params.inputs[1], params.input1_order);
        TransposedDimensionAccessHelperBase output(params.outputs[0], params.output_order);

        const size_t batch_size = output.b_dim().v;
        const size_t heads_num = output.f_dim().v;
        const size_t source_seq_len = dims_k.y_dim().v;
        const size_t target_seq_len = dims_q.y_dim().v;
        const size_t head_size = static_cast<size_t>(params.conf.head_size);
        const size_t num_of_partitions = CeilDiv(source_seq_len, get_seq_len_partition_size());
        const size_t target_seq_len_block_size = kernel_idx == 1 ? get_target_seq_len_block_size() : 1;

        if (kernel_idx == KernelsTypes::SINGLE_TOKEN || kernel_idx == KernelsTypes::MULTI_TOKENS) {
            dispatch_data.gws = { batch_size * heads_num,
                                  CeilDiv(target_seq_len, target_seq_len_block_size),
                                  head_size * num_of_partitions };
            dispatch_data.lws = { 1, 1, head_size };
        } else if (kernel_idx == 2) {
            dispatch_data.gws = { batch_size * heads_num,
                                  target_seq_len,
                                  16 };
            dispatch_data.lws = { 1, 1, 16 };
        }
    }

    return dispatch_data;
}

KernelsData SDPAKernelOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    // Implementation contains multiple kernels:
    // kernel[0] - single token generation stage (2nd token)
    // kernel[1] - multi tokens processing stage (1st token)
    // kernel[2] - results aggregation

    const size_t kernels_num = KernelsTypes::TOTAL_KERNELS_NUM;
    KernelData kd = KernelData::Default<sdpa_params>(params, kernels_num);
    kd.needs_sub_kernels_sync = true;

    GetUpdateDispatchDataFunc(kd);

    const auto& prim_params = dynamic_cast<const sdpa_params&>(params);
    for (size_t kernel_idx = 0; kernel_idx < kernels_num; kernel_idx++) {
        auto dispatch_data = SetDefault(prim_params, kernel_idx);
        auto kernel_name = kernel_idx == 0 ? kernelName + "_single_token" :
                                             kernel_idx == 1 ? kernelName + "_multi_tokens" : kernelName + "_finalization";
        auto entry_point = GetEntryPoint(kernel_name, prim_params.layerID, params);
        auto jit_constants = GetJitConstants(prim_params, kernel_idx);
        auto jit = CreateJit(kernel_name, jit_constants, entry_point);

        auto& kernel = kd.kernels[kernel_idx];

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

        const auto num_of_partitions = 1;
        auto& output = prim_params.outputs[0];
        auto head_size = output.X().v;

        auto buf_dt_size = 4;
        auto buf_elements_count = (num_of_partitions == 1) ? 1 : output.LogicalSize() / head_size * num_of_partitions;
        auto buf_size = buf_elements_count * buf_dt_size;

        auto tmp_out_dt_size = 4;
        auto tmp_out_elements_count = (num_of_partitions == 1) ? 1 : output.LogicalSize() * num_of_partitions;
        auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});

        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(buf_size);
        kd.internalBufferSizes.push_back(buf_size);
        kd.internalBufferSizes.push_back(tmp_out_size);
        kd.internalBufferDataType = prim_params.inputs[0].GetDType();

        if (kernel_idx == KernelsTypes::FINALIZATION) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

            ScalarDescriptor num_of_partitions_scalar;
            num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
            num_of_partitions_scalar.v.u32 = num_of_partitions;

            kernel.params.scalars.clear();
            kernel.params.scalars.push_back(num_of_partitions_scalar);
        }
    }

    return { kd };
}

void SDPAKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kernel_data) {
        const auto& prim_params = static_cast<const sdpa_params&>(params);

        const size_t expected_kernels_num = KernelsTypes::TOTAL_KERNELS_NUM;
        OPENVINO_ASSERT(kernel_data.kernels.size() == expected_kernels_num,
                        "[GPU] Invalid kernels size for update dispatch data func of SDPA kernel");

        TransposedDimensionAccessHelperBase dims_q(prim_params.inputs[0], prim_params.input0_order);
        TransposedDimensionAccessHelperBase dims_k(prim_params.inputs[1], prim_params.input1_order);
        auto& output = prim_params.outputs[0];

        auto target_seq_len = dims_q.y_dim().v;
        auto head_size = dims_q.x_dim().v;
        auto source_seq_len = dims_k.y_dim().v;

        auto num_of_partitions = CeilDiv(source_seq_len, get_seq_len_partition_size());

        auto buf_dt_size = output.ElementSize();
        auto buf_elements_count = (num_of_partitions == 1) ? 1 : output.LogicalSize() / head_size * num_of_partitions;
        auto buf_size = buf_elements_count * buf_dt_size;

        auto tmp_out_dt_size = output.ElementSize();
        auto tmp_out_elements_count = (num_of_partitions == 1) ? 1 : output.LogicalSize() * num_of_partitions;
        auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

        auto dispatch_data1 = SetDefault(prim_params, 0);
        kernel_data.kernels[0].params.workGroups.global = dispatch_data1.gws;
        kernel_data.kernels[0].params.workGroups.local = dispatch_data1.lws;
        kernel_data.kernels[0].skip_execution = target_seq_len > 1;

        auto dispatch_data2 = SetDefault(prim_params, 1);
        kernel_data.kernels[1].params.workGroups.global = dispatch_data2.gws;
        kernel_data.kernels[1].params.workGroups.local = dispatch_data2.lws;
        kernel_data.kernels[1].skip_execution = target_seq_len == 1;

        ScalarDescriptor num_of_partitions_scalar;
        num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
        num_of_partitions_scalar.v.u32 = num_of_partitions;

        auto dispatch_data3 = SetDefault(prim_params, 2);
        kernel_data.kernels[2].params.workGroups.global = dispatch_data3.gws;
        kernel_data.kernels[2].params.workGroups.local = dispatch_data3.lws;
        kernel_data.kernels[2].skip_execution = num_of_partitions == 1;

        kernel_data.kernels[2].params.scalars.clear();
        kernel_data.kernels[2].params.scalars.push_back(num_of_partitions_scalar);

        kernel_data.internalBufferSizes.clear();
        kernel_data.internalBufferSizes.push_back(buf_size);
        kernel_data.internalBufferSizes.push_back(buf_size);
        kernel_data.internalBufferSizes.push_back(tmp_out_size);
        kernel_data.internalBufferDataType = prim_params.inputs[0].GetDType();
    };
}

KernelsPriority SDPAKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}
}  // namespace kernel_selector
