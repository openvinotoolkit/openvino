// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_ref.hpp"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

// For kernel w/o split
constexpr size_t max_sequence_length = 3072;

constexpr size_t seq_len_portion_size = 256;
constexpr size_t subgroup_size = 16;

const Datatype softmax_acc_dt = Datatype::F32;

// Use flash attention or not
const bool use_seq_len_split = true;

template <typename T>
T convert_to(const std::string &str) {
    std::istringstream ss(str);
    T res;
    ss >> res;
    return res;
}

template <>
std::string convert_to(const std::string &str) {
    return str;
}

void SDPAKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const sdpa_params&>(params);
        bool use_seq_len_split = false;
        if (const auto env_var = std::getenv("USE_SPLIT")) {
            use_seq_len_split = convert_to<bool>(env_var);
        }
        const size_t expected_kernels_num = use_seq_len_split ? 2 : 1;
        OPENVINO_ASSERT(kd.kernels.size() == expected_kernels_num, "[GPU] Invalid kernels size for update dispatch data func of SDPA kernel");

        auto dispatchData = SetDefault(prim_params, 0);
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;

        if (expected_kernels_num == 2) {
            const auto& input = prim_params.inputs[0];
            const size_t batch_size = input.Batch().v;
            const size_t seq_len = input.Feature().v;
            const size_t tokens_num = batch_size * seq_len;
            const size_t num_of_portions = CeilDiv(prim_params.configuration.max_context_len, seq_len_portion_size);

            auto dispatchData = SetDefault(prim_params, 1);
            kd.kernels[1].params.workGroups.global = dispatchData.gws;
            kd.kernels[1].params.workGroups.local = dispatchData.lws;
            // Write directly to the output in SDPA main kernel in case of single portion
            kd.kernels[1].skip_execution = num_of_portions == 1;

            auto buf_dt_size = 4;
            auto buf_elements_count = tokens_num * prim_params.configuration.heads_num * num_of_portions;
            auto buf_size = buf_elements_count * buf_dt_size;

            auto tmp_out_dt_size = 4;
            auto tmp_out_elements_count = tokens_num * prim_params.configuration.heads_num * num_of_portions * prim_params.configuration.head_size;
            auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

            kd.internalBufferSizes.clear();
            kd.internalBufferSizes.push_back(buf_size);
            kd.internalBufferSizes.push_back(buf_size);
            kd.internalBufferSizes.push_back(tmp_out_size);
            kd.internalBufferDataType = softmax_acc_dt;

            ScalarDescriptor num_of_portions_scalar;
            num_of_portions_scalar.t = ScalarDescriptor::Types::UINT32;
            num_of_portions_scalar.v.u32 = num_of_portions;

            kd.kernels[1].params.scalars.resize(1);
            kd.kernels[1].params.scalars[0] = num_of_portions_scalar;
        }
    };
}

KernelsData SDPAKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }
    bool use_seq_len_split = false;
    if (const auto env_var = std::getenv("USE_SPLIT")) {
        use_seq_len_split = convert_to<bool>(env_var);
        static bool printed = false;
        if (!printed)
            std::cout << "Force SPLIT to " << use_seq_len_split << "\n";
        printed = true;
    }
    const uint kernels_num = use_seq_len_split ? 2 : 1;
    KernelData kd = KernelData::Default<sdpa_params>(params, kernels_num);
    kd.needs_sub_kernels_sync = true;
    GetUpdateDispatchDataFunc(kd);

    const auto& kernel_params = static_cast<const sdpa_params&>(params);

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

            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
            kd.internalBufferSizes.clear();
            kd.internalBufferSizes.push_back(1);
            kd.internalBufferSizes.push_back(1);
            kd.internalBufferSizes.push_back(1);
            kd.internalBufferDataType = softmax_acc_dt;

            if (i == 1) {
                ScalarDescriptor block_elem_num;
                block_elem_num.t = ScalarDescriptor::Types::UINT32;
                block_elem_num.v.u32 = 0;
                kernel.params.scalars.push_back(block_elem_num);
                kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
            }
        }
    }

    return {kd};
}

ParamsKey SDPAKernelRef::GetSupportedKey() const {
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

bool SDPAKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::PA_SDPA) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const sdpa_params&>(params);
    if (seq_len_portion_size % kernel_params.configuration.block_size != 0)
        return false;

    if (kernel_params.configuration.head_size % subgroup_size != 0)
        return false;

    const auto subgroups_per_wg = kernel_params.configuration.head_size / subgroup_size;
    if (subgroups_per_wg > subgroup_size)
        return false;

    return true;
}

JitConstants SDPAKernelRef::GetJitConstants(const sdpa_params& kernel_params) const {
    JitConstants jit = MakeBaseParamsJitConstants(kernel_params);

    const auto& config = kernel_params.configuration;
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", config.head_size));
    jit.AddConstant(MakeJitConstant("HEADS_NUM", config.heads_num));
    jit.AddConstant(MakeJitConstant("KV_HEADS_NUM", config.kv_heads_num));
    jit.AddConstant(MakeJitConstant("NUM_QUERIES_PER_KV_HEAD", config.heads_num / config.kv_heads_num));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", config.block_size));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", config.x_block_size));
    jit.Merge(MakeTypeJitConstants(softmax_acc_dt, "ACCUMULATOR"));

    bool use_seq_len_split = false;
    if (const auto env_var = std::getenv("USE_SPLIT")) {
        use_seq_len_split = convert_to<bool>(env_var);
    }

    if (use_seq_len_split) {
        jit.AddConstant(MakeJitConstant("USE_SEQ_LEN_SPLIT", true));
        jit.AddConstant(MakeJitConstant("SEQ_LEN_PORTION_SIZE", seq_len_portion_size));
        jit.AddConstant(MakeJitConstant("SHARED_MEM_SIZE", seq_len_portion_size));
    } else {
        jit.AddConstant(MakeJitConstant("SHARED_MEM_SIZE", max_sequence_length));
    }

    return jit;
}

CommonDispatchData SDPAKernelRef::SetDefault(const sdpa_params& kernel_params, size_t kernel_idx) {
    CommonDispatchData dispatch_data;

    bool use_seq_len_split = false;
    if (const auto env_var = std::getenv("USE_SPLIT")) {
        use_seq_len_split = convert_to<bool>(env_var);
    }

    const auto& input = kernel_params.inputs[0];
    if (!input.is_dynamic()) {
        const size_t batch_size = input.Batch().v;
        const size_t seq_len = input.Feature().v;
        const size_t tokens_num = batch_size * seq_len;

        const size_t num_of_portions =
            use_seq_len_split ? CeilDiv(kernel_params.configuration.max_context_len, seq_len_portion_size) : 1;

        if (kernel_idx == 0) {
            dispatch_data.gws = { tokens_num,
                                  kernel_params.configuration.heads_num,
                                  kernel_params.configuration.head_size * num_of_portions };
            dispatch_data.lws = { 1, 1, kernel_params.configuration.head_size };
        } else {
            dispatch_data.gws = { batch_size,
                                  seq_len,
                                  kernel_params.configuration.head_size * kernel_params.configuration.heads_num };
            dispatch_data.lws = { 1, 1, subgroup_size };
        }
    }

    return dispatch_data;
}

}  // namespace kernel_selector
