// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_base.h"
#include "kernel_selector_utils.h"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace kernel_selector {

static std::string GetDimsOrder(const std::vector<int64_t>& order_idx) {
    auto get_order_idx = [](std::vector<int64_t> order_idx, int64_t dim_idx) {
        int loc = 0;
        for (auto idx : order_idx) {
            if (idx == dim_idx)
                break;
            loc += 1;
        }
        return loc;
    };

    std::string dims_order = "";
    if (order_idx.size() == 2) {
        const std::vector<std::string> dims2 = {"y", "x"};
        dims_order = "b,f,w,z,"
                    + dims2[get_order_idx(order_idx, 0)] + "," + dims2[get_order_idx(order_idx, 1)];
    } else if (order_idx.size() == 3) {
        const std::vector<std::string> dims3 = {"f", "y", "x"};
        dims_order = "b," + dims3[get_order_idx(order_idx, 0)] + ",w,z,"
                    + dims3[get_order_idx(order_idx, 1)] + "," + dims3[get_order_idx(order_idx, 2)];
    } else if (order_idx.size() == 4) {
        const std::vector<std::string> dims4 = {"b", "f", "y", "x"};
        dims_order = dims4[get_order_idx(order_idx, 0)] + "," + dims4[get_order_idx(order_idx, 1)] + ",w,z,"
                    + dims4[get_order_idx(order_idx, 2)] + "," + dims4[get_order_idx(order_idx, 3)];
    } else if (order_idx.size() == 5) {
        const std::vector<std::string> dims5 = {"b", "f", "z", "y", "x"};
        dims_order = dims5[get_order_idx(order_idx, 0)] + "," + dims5[get_order_idx(order_idx, 1)] + ",w,"
                    + dims5[get_order_idx(order_idx, 2)] + "," + dims5[get_order_idx(order_idx, 3)] + ","
                    + dims5[get_order_idx(order_idx, 4)];
    } else if (order_idx.size() == 6) {
        const std::vector<std::string> dims6 = {"b", "f", "w", "z", "y", "x"};
        dims_order = dims6[get_order_idx(order_idx, 0)] + "," + dims6[get_order_idx(order_idx, 1)] + ","
                    + dims6[get_order_idx(order_idx, 2)] + "," + dims6[get_order_idx(order_idx, 3)] + ","
                    + dims6[get_order_idx(order_idx, 4)] + "," + dims6[get_order_idx(order_idx, 5)];
    } else {
        dims_order = "b,f,w,z,y,x";
    }
    return dims_order;
}

static std::string GetBroadcastInputStr(const size_t input_rank, const int64_t axes, const int64_t val) {
    std::vector<std::string> dims;
    if (input_rank == 1) {
        dims = {"x"};
    } else if (input_rank == 2) {
        dims = {"y", "x"};
    } else if (input_rank == 3) {
        dims = {"f", "y", "x"};
    } else if (input_rank == 4) {
        dims = {"b", "f", "y", "x"};
    } else if (input_rank == 5) {
        dims = {"b", "f", "z", "y", "x"};
    } else if (input_rank == 6) {
        dims = {"b", "f", "w", "z", "y", "x"};
    }
    return dims[axes] + " /= " + std::to_string(val) + ";";
}

JitConstants SDPAKernelBase::GetJitConstants(const sdpa_params& params) const {
    auto jit = MakeBaseParamsJitConstants(params);

    if (params.conf.broadcast_axis != -1) {
        jit.AddConstant(MakeJitConstant("BROADCAST_GROUP_SIZE", params.conf.group_size));
        jit.AddConstant(MakeJitConstant("DO_BROADCAST_KEY_VALUE", GetBroadcastInputStr(params.inputs[0].GetDims().size(),
                                                                                       params.conf.broadcast_axis,
                                                                                       params.conf.group_size)));
    } else {
        jit.AddConstant(MakeJitConstant("BROADCAST_GROUP_SIZE", 1));
    }

    jit.AddConstant(MakeJitConstant("IS_CAUSAL", params.conf.is_causal));
    if (!params.conf.is_paged_attention) {
        jit.AddConstant(MakeJitConstant("HAS_ATTN_MASK_INPUT", params.inputs.size() > 3));
        jit.AddConstant(MakeJitConstant("HAS_SCALE_INPUT", params.inputs.size() > 4));
    }

    jit.AddConstant(MakeJitConstant("IS_KV_COMPRESSED", params.conf.is_kv_compressed));

    if (params.conf.is_kv_compressed) {
        jit.AddConstant(MakeJitConstant("USE_ASYMMETRIC_QUANTIZATION", params.conf.use_asymmetric_quantization));
        jit.AddConstant(MakeJitConstant("COMBINE_SCALES_AND_ZP", params.conf.combine_scales_and_zp));
        jit.AddConstant(MakeJitConstant("COMPRESSED_PER_HEAD", params.conf.per_head_quantization));
        jit.AddConstant(MakeJitConstant("KEY_COMPRESSION_SCALE", params.key_cache_comp_scale));
        jit.AddConstant(MakeJitConstant("VALUE_COMPRESSION_SCALE", params.value_cache_comp_scale));

        if (params.conf.use_asymmetric_quantization && !params.conf.combine_scales_and_zp) {
            jit.AddConstant(MakeJitConstant("KEY_COMPRESSION_ZP", params.key_cache_comp_zp));
            jit.AddConstant(MakeJitConstant("VALUE_COMPRESSION_ZP", params.value_cache_comp_zp));
        }
    }

    auto is_default_order = [](const std::vector<int64_t>& order) {
        for (size_t i = 0; i < order.size(); i++)
            if (order[i] != static_cast<int64_t>(i))
                return false;
        return true;
    };

    auto use_index_calc_func = [&](const std::vector<int64_t> order, bool is_query = false) {
        if (!params.input0_order.empty() && !is_default_order(params.input0_order))
            return true;

        if (params.conf.broadcast_axis != -1)
            return true;

        if (params.indirect_axis != -1 && !is_query)
            return true;

        return false;
    };

    if (params.indirect_axis != -1)
        jit.AddConstant(MakeJitConstant("BEAM_TABLE", params.beam_table));

    if (use_index_calc_func(params.input0_order, true))
        jit.AddConstant(MakeJitConstant("INPUT0_DIMS_ORDER", GetDimsOrder(params.input0_order)));

    if (use_index_calc_func(params.input1_order))
        jit.AddConstant(MakeJitConstant("INPUT1_DIMS_ORDER", GetDimsOrder(params.input1_order)));

    if (use_index_calc_func(params.input2_order))
        jit.AddConstant(MakeJitConstant("INPUT2_DIMS_ORDER", GetDimsOrder(params.input2_order)));

    TransposedDimensionAccessHelperJit dims_q(params.inputs[0], params.input0_order);
    const auto num_heads = params.conf.is_paged_attention ? std::to_string(params.conf.heads_num) : dims_q.f();
    jit.AddConstant(MakeJitConstant("TARGET_SEQ_LEN", dims_q.y()));
    jit.AddConstant(MakeJitConstant("NUM_HEADS", num_heads));
    jit.AddConstant(MakeJitConstant("NUM_KV_HEADS", params.conf.kv_heads_num));

    TransposedDimensionAccessHelperJit dims_k(params.inputs[1], params.input1_order);
    jit.AddConstant(MakeJitConstant("SOURCE_SEQ_LEN", dims_k.y()));

    return jit;
}

bool SDPAKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SDPA) {
        return false;
    }

    const sdpa_params& params = static_cast<const sdpa_params&>(p);

    for (size_t i = 0; i < params.inputs.size(); i++) {
        if (params.inputs[i].Dimentions() != 4)
            return false;
    }

    if (params.outputs[0].Dimentions() != 4)
        return false;

    return true;
}
}  // namespace kernel_selector
