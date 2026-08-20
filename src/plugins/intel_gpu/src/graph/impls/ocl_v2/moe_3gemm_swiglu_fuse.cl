// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if GATHER_ENABLE
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (gather_2d_ref)(
    const __global MOE_DTYPE* src_tok,       // input tokens [total_token, hidden_size] - hidden_states_mem_ptr
    const __global MOE_DTYPE* src_rweight,   // topk_weights [total_token, topk_experts]
    __global int * tok_index,                // token index [expert_idx][] = [actual_token_num]   - expert_mask_mem.batch
    __global int * top_index,                // topk  index [expert_idx][] = [actual_token_num]   - expert_mask_mem.topk
    __global MOE_DTYPE* dst_tok,             // output tokens [batch_size, hidden_size] - scratch.x
    __global MOE_DTYPE* dst_rweight) {       // output topk_weights [batch_size] - scratch.routing_weights

    int k = get_global_id(0);   // token_idx
    int off = get_global_id(1); // hidden_size offset
    int tok_idx = tok_index[k];

    src_tok += tok_idx * HIDDEN_SIZE;
    dst_tok += k * HIDDEN_SIZE;

    if (off >= HIDDEN_SIZE) {
        // printf("Warning off >= HIDDEN_SIZE: k = %d, off = %d, HIDDEN_SIZE = %d\n", k, off, HIDDEN_SIZE);
        return;
    }

    #if MOE_DTYPE_SIZE == 2
        ushort value = intel_sub_group_block_read_us((const __global ushort *)(src_tok + off));
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), value);
    #elif MOE_DTYPE_SIZE == 4
        uint value = intel_sub_group_block_read((const __global uint *)(src_tok + off));
        intel_sub_group_block_write((__global uint *)(dst_tok + off), value);
    #else
        dst_tok[off] = src_tok[off];
    #endif

    if (off == 0) {
        int top_idx = top_index[k];
        dst_rweight[k] = src_rweight[top_idx];
    }
}

#elif SCATTER_ENABLE
KERNEL (index_add_)(const __global MOE_DTYPE* src_tok,
    __global int * tok_index,
    __global MOE_DTYPE* dst_tok) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];

    src_tok += k * HIDDEN_SIZE;
    dst_tok += tok_idx * HIDDEN_SIZE;

    #if MOE_DTYPE_SIZE == 2
        half src_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(src_tok + off)));
        half dst_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(dst_tok + off)));
        half value = dst_value + src_value;
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), as_ushort(value));
    #elif MOE_DTYPE_SIZE == 4
        float src_value = as_float(intel_sub_group_block_read((const __global uint *)(src_tok + off)));
        float dst_value = as_float(intel_sub_group_block_read((const __global uint *)(dst_tok + off)));
        float value = dst_value + src_value;
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), as_uint(value));
    #else
        dst_tok[off] += src_tok[off];
    #endif
}

#elif PREFILL_SWIGLU_ENABLE

#define SWISH_BETA 1.0f
#define ACC_DTYPE float

// Tanh-approximation Gelu: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#define GELU_TANH_SQRT_2_OVER_PI 0.7978845608028654f
#define GELU_TANH_C 0.044715f

// ERF Gelu: 0.5 * x * (1 + erf(x / sqrt(2))); 1/sqrt(2) = 0.7071067811865475
// Fast erf approximation (A&S 7.1.26) — same coefficients as swiglu_gpu_opt.cl
inline float moe_fast_erf(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    float z = fabs(x);
    float t = 1.0f / (1.0f + p * z);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * native_exp(-(z * z));
    return (x >= 0.0f) ? y : -y;
}

inline ACC_DTYPE moe_gate_activation(ACC_DTYPE x) {
#if GATE_ACT_GELU_ERF
    return 0.5f * x * (1.0f + moe_fast_erf(x * 0.7071067811865475f));
#elif GATE_ACT_GELU_TANH
    return 0.5f * x * (1.0f + (tanh(0.79788458347320556640625f * x * (1.0f + 0.044715f * x * x))));
#else
    // Swish (SwiGLU): x * sigmoid(beta * x)
    return x / (1.0f + native_exp(-SWISH_BETA * x));
#endif
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(swiglu_ref) (
    const __global MOE_DTYPE* up, // [token_len * expert_topK, inter_size]
    const __global MOE_DTYPE* gate,
    __global MOE_DTYPE* output    // [token_len * expert_topK, inter_size]
) {
    const uint token_idx = get_global_id(1);
    const uint n_offset = get_global_id(0);
    // gws = {_intermediate_size, token_cnt,  1}
    // lws = {subgroup_size, 1, 1};

#if MOE_DTYPE_SIZE == 2
    const uint sg_id = get_sub_group_local_id();
    const uint offset = token_idx * INTERMEDIA_SIZE + n_offset - sg_id;
    ACC_DTYPE up_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(up + offset)));
    ACC_DTYPE gate_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(gate + offset)));
    ACC_DTYPE value = moe_gate_activation(gate_value);
    half result = value * up_value;
    intel_sub_group_block_write_us((__global ushort *)(output + offset), as_ushort(result));
#else
    const uint offset = token_idx * INTERMEDIA_SIZE + n_offset;
    ACC_DTYPE gate_value = gate[offset];
    ACC_DTYPE up_value = up[offset];
    ACC_DTYPE value = moe_gate_activation(gate_value);
    ACC_DTYPE result = value * up_value;
    output[offset] = result;
#endif
}

#endif
