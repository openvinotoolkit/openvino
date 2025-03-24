// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
struct TransposedDimensionAccessHelperBase : virtual DimensionAccessHelperBase {
    explicit TransposedDimensionAccessHelperBase(const DataTensor& t, std::vector<int64_t> order)
    : DimensionAccessHelperBase(t) {
        size_t total_dims_count = dims.size();
        size_t new_axis_count = total_dims_count - order.size();

        transposed_order.resize(total_dims_count);
        std::iota(transposed_order.begin(), transposed_order.end(), 0);
        for (size_t i = 0; i < order.size(); i++) {
            size_t transposed_order_pos = i < 2 ? i : i + new_axis_count;
            transposed_order[transposed_order_pos] = order[i] < 2 ? order[i] : order[i] + new_axis_count;
        }
    }

    Tensor::Dim& x_dim() { return dims[transposed_order[7]]; }
    Tensor::Dim& y_dim() { return dims[transposed_order[6]]; }
    Tensor::Dim& z_dim() { return dims[transposed_order[5]]; }
    Tensor::Dim& w_dim() { return dims[transposed_order[4]]; }
    Tensor::Dim& v_dim() { return dims[transposed_order[3]]; }
    Tensor::Dim& u_dim() { return dims[transposed_order[2]]; }
    Tensor::Dim& f_dim() { return dims[transposed_order[1]]; }
    Tensor::Dim& b_dim() { return dims[transposed_order[0]]; }

    std::vector<int64_t> transposed_order;
};

struct TransposedDimensionAccessHelperJit : DimensionAccessHelperJit, TransposedDimensionAccessHelperBase {
    explicit TransposedDimensionAccessHelperJit(const DataTensor& t, std::vector<int64_t> order, bool padded = false)
    : DimensionAccessHelperBase(t)
    , DimensionAccessHelperJit(t, padded)
    , TransposedDimensionAccessHelperBase(t, order) {}

    std::string x() { return dims_sizes[transposed_order[7]]; }
    std::string y() { return dims_sizes[transposed_order[6]]; }
    std::string z() { return dims_sizes[transposed_order[5]]; }
    std::string w() { return dims_sizes[transposed_order[4]]; }
    std::string v() { return dims_sizes[transposed_order[3]]; }
    std::string u() { return dims_sizes[transposed_order[2]]; }
    std::string f() { return dims_sizes[transposed_order[1]]; }
    std::string b() { return dims_sizes[transposed_order[0]]; }

    std::pair<std::string, std::string> x_pad() {
        return {pad_before_after_sizes[(transposed_order[7] * 2) + 0], pad_before_after_sizes[(transposed_order[7] * 2) + 1]};
    }
    std::pair<std::string, std::string> y_pad() {
        return {pad_before_after_sizes[(transposed_order[6] * 2) + 0], pad_before_after_sizes[(transposed_order[6] * 2) + 1]};
    }
    std::pair<std::string, std::string> z_pad() {
        return {pad_before_after_sizes[(transposed_order[5] * 2) + 0], pad_before_after_sizes[(transposed_order[5] * 2) + 1]};
    }
    std::pair<std::string, std::string> w_pad() {
        return {pad_before_after_sizes[(transposed_order[4] * 2) + 0], pad_before_after_sizes[(transposed_order[4] * 2) + 1]};
    }
    std::pair<std::string, std::string> v_pad() {
        return {pad_before_after_sizes[(transposed_order[3] * 2) + 0], pad_before_after_sizes[(transposed_order[3] * 2) + 1]};
    }
    std::pair<std::string, std::string> u_pad() {
        return {pad_before_after_sizes[(transposed_order[2] * 2) + 0], pad_before_after_sizes[(transposed_order[2] * 2) + 1]};
    }
    std::pair<std::string, std::string> f_pad() {
        return {pad_before_after_sizes[(transposed_order[1] * 2) + 0], pad_before_after_sizes[(transposed_order[1] * 2) + 1]};
    }
    std::pair<std::string, std::string> b_pad() {
        return {pad_before_after_sizes[(transposed_order[0] * 2) + 0], pad_before_after_sizes[(transposed_order[0] * 2) + 1]};
    }
};

struct sdpa_configuration {
    int64_t head_size = -1;
    int64_t heads_num = -1;
    int64_t kv_heads_num = -1;

    // GQA configuration
    int64_t kv_group_size = 1;
    int64_t broadcast_axis = -1;

    bool is_causal = false;
    bool has_alibi_input = false;
    bool is_kv_compressed = false;
    bool use_asymmetric_quantization = false;
    bool combine_scales_and_zp = false;
    bool per_head_quantization = false;

    // Paged Attention configuration
    bool is_paged_attention = false;
    size_t paged_attention_sliding_window = 0;
    int64_t paged_attention_aligned_seq_len = -1;
    int64_t paged_attention_block_size = 0;
    int64_t paged_attention_max_len = 0;
    bool has_const_scale_val = false;
    float scale_val = 0.f;
    bool has_rotated_blocks = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// sdpa_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct sdpa_params : public base_params {
    sdpa_params() : base_params(KernelType::SDPA) {}

    std::vector<int64_t> input0_order;
    std::vector<int64_t> input1_order;
    std::vector<int64_t> input2_order;
    std::vector<int64_t> output_order;
    int64_t indirect_axis = -1;

    DataTensor beam_table;
    DataTensor key_cache_comp_scale;
    DataTensor key_cache_comp_zp;
    DataTensor value_cache_comp_scale;
    DataTensor value_cache_comp_zp;

    sdpa_configuration conf;
    bool should_use_sdpa_opt = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SDPAKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SDPAKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~SDPAKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const sdpa_params& params) const;
};
}  // namespace kernel_selector
