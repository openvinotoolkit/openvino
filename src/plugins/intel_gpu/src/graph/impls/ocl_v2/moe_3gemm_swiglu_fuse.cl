// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if SOFTMAX_TOPK_ENABLE

KERNEL(softmax_topk)(
    const __global MOE_DTYPE* input, // [input_batch, sort_in_num]
    __global uint* output_index, // [input_batch, TOP_K]
    __global MOE_DTYPE* output // [input_batch, TOP_K]
) {
    // gws [batch, sort_in_num]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);

    input += batch * sort_cnt + sort_index;

    uint sort_position = 0;

    __local MOE_DTYPE local_input[VALUE_NUM];
    __local MOE_DTYPE local_output[TOP_K];
    __local uint local_index[TOP_K];

    MOE_DTYPE in_value = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
    local_input[sort_index] = in_value;
    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_index; i++) {
        MOE_DTYPE value = local_input[i];
        if(value >= in_value) {
            sort_position++;
        }
    }

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = sort_index; i < sort_cnt; i++) {
        MOE_DTYPE value = local_input[i];
        if(value > in_value) {
            sort_position++;
        }
    }
    if (sort_position < TOP_K) {
        local_output[sort_position] = in_value;
        local_index[sort_position] = sort_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(sort_position == 0) {
        float softmax_total = 1.0;
        MOE_DTYPE max_v = local_output[0];
        local_output[0] = 1;
        for(uint i = 1; i < TOP_K; i++) {
            local_output[i] = native_exp(local_output[i] - max_v);
            softmax_total += local_output[i];
        }
        output_index += batch * TOP_K;
        output += batch * TOP_K;

        for(uint i = 0; i < TOP_K; i++) {
            output[i] = local_output[i]/softmax_total;
            output_index[i] = local_index[i];
        }
    }
}

#elif GATHER_ENABLE
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
        printf("Warning off >= HIDDEN_SIZE: k = %d, off = %d, HIDDEN_SIZE = %d\n", k, off, HIDDEN_SIZE);
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
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(swiglu_ref) (
    const __global MOE_DTYPE* up, // [token_len * expert_topK, hidden_size]
    const __global MOE_DTYPE* gate,
    __global MOE_DTYPE* output    // [token_len * expert_topK, hidden_size]
) {
    const uint token_idx = get_global_id(0);
    const uint n_offset = get_global_id(1);

    const uint offset = token_idx * INTERMEDIA_SIZE + n_offset;
#if MOE_DTYPE_SIZE == 2
    half up_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(up + offset)));
    half gate_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(gate + offset)));
    half value = gate_value / (1.0 + native_exp(-SWISH_BETA * gate_value));
    MOE_DTYPE result = value * up_value;
    intel_sub_group_block_write_us((__global ushort *)(output + offset), as_ushort(result));
#else
    MOE_DTYPE gate_value = gate[offset];
    MOE_DTYPE up_value = up[offset];
    half value = gate_value / (1.0 + native_exp(-SWISH_BETA * gate_value));
    MOE_DTYPE result = value * up_value;
    output[offset] = result;
#endif
}

#elif PREFILL_SCALE_ZP_REPACK
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(repack_ref) (
    const __global half* scale_src,
    const __global char* zp_src,
    __global half* scale_dst,
    __global char* zp_dst
) {
    const uint expert_idx = get_global_id(0);
    const uint group_num = get_global_size(1);
    const uint n_num = get_global_size(2) * 2;
    const uint group_idx = get_global_id(1);
    const uint n_idx = get_global_id(2) * 2;

    // Source: [Experts, Groups, N]
    // Dest:   [Experts, N, Groups]

#if 0
    const uint src_offset = expert_idx * n_num * group_num + group_idx * n_num + n_idx;
    const uint dst_offset = expert_idx * n_num * group_num + n_idx * group_num + group_idx;

    scale_src += src_offset;
    zp_src += src_offset / 2;
    scale_dst += dst_offset;
    zp_dst += dst_offset / 2;

    half2 src_value = as_half2(intel_sub_group_block_read_us2((const __global ushort *)(scale_src)));
    intel_sub_group_block_write_us2((__global ushort *)(scale_dst), as_ushort2(src_value));

    char zp_value = zp_src[0];
    zp_dst[0] = as_uchar(zp_value);
#else
    // Calculate offsets for Scale (Source)
    // src index for (e, g, n)
    const uint src_idx_0 = expert_idx * group_num * n_num + group_idx * n_num + n_idx;
    
    // Read Scale
    half s0 = scale_src[src_idx_0];
    half s1 = scale_src[src_idx_0 + 1];

    // Calculate offsets for Scale (Dest)
    // dst index for (e, n, g)
    const uint dst_idx_0 = expert_idx * n_num * group_num + n_idx * group_num + group_idx;
    // dst index for (e, n+1, g)
    const uint dst_idx_1 = expert_idx * n_num * group_num + (n_idx + 1) * group_num + group_idx;

    // Write Scale
    scale_dst[dst_idx_0] = s0;
    scale_dst[dst_idx_1] = s1;

    // Handle ZP
    // Only even groups process ZP to avoid race condition on byte writes
    if (group_idx % 2 == 0) {
        // We need zp(g, n), zp(g, n+1) -> from src(e, g, n/2)
        // We need zp(g+1, n), zp(g+1, n+1) -> from src(e, g+1, n/2)
        
        // Src ZP index for (e, g, n/2)
        // Note: zp_src is char*, so index is byte index.
        // src_idx_0 is element index. ZP packed 2 elements per byte.
        uint zp_src_idx_g0 = src_idx_0 / 2;
        
        // Src ZP index for (e, g+1, n/2)
        // Offset difference between g and g+1 is n_num elements.
        uint zp_src_idx_g1 = zp_src_idx_g0 + (n_num / 2); 
        
        char byte_g0 = zp_src[zp_src_idx_g0];
        char byte_g1 = zp_src[zp_src_idx_g1];
        
        // Unpack
        // Assuming Little Endian packing: Low nibble = even index (n), High nibble = odd index (n+1)
        uchar ubyte_g0 = as_uchar(byte_g0);
        uchar ubyte_g1 = as_uchar(byte_g1);
        
        uchar zp_g0_n0 = ubyte_g0 & 0x0F;
        uchar zp_g0_n1 = (ubyte_g0 >> 4) & 0x0F;
        
        uchar zp_g1_n0 = ubyte_g1 & 0x0F;
        uchar zp_g1_n1 = (ubyte_g1 >> 4) & 0x0F;
        
        // Pack for Dest
        // Dest packing is along Groups.
        // Byte at (e, n, g/2) contains zp(e, n, g) [Low] and zp(e, n, g+1) [High]
        
        uchar dst_byte_n0 = zp_g0_n0 | (zp_g1_n0 << 4);
        uchar dst_byte_n1 = zp_g0_n1 | (zp_g1_n1 << 4);

        // Dest ZP indices
        // dst_idx_0 corresponds to (e, n, g).
        // ZP array is packed along G.
        // So index is dst_idx_0 / 2.
        uint zp_dst_idx_n0 = dst_idx_0 / 2;
        
        // dst_idx_1 corresponds to (e, n+1, g).
        uint zp_dst_idx_n1 = dst_idx_1 / 2;
        
        zp_dst[zp_dst_idx_n0] = as_char(dst_byte_n0);
        zp_dst[zp_dst_idx_n1] = as_char(dst_byte_n1);
    }
#endif
}
#endif
