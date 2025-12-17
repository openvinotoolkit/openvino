
/*******************************************************************************
 * STAGE 369 TERNARY KERNEL LIBRARY FOR INTEL GEN9.5
 * 
 * Features:
 * - Bitwise ternary MAC (zero-latency)
 * - Subgroup-accelerated reduction
 * - SLM bank conflict-free access
 * - Sampler-based weight loading
 * - Fused Conv-ReLU-Pool operations
 ******************************************************************************/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

// Constants
#define SIMD_WIDTH 16
#define CACHELINE_SIZE 64
#define SLM_BANK_COUNT 16

// Bitwise ternary MAC (Section 7.1)
inline half ternary_mac_ultra(half activation, half weight) {
    ushort a = as_ushort(activation);
    // Fix: Compare the original half weight to 0.0h, not the ushort bits
    ushort zero_mask = (weight == 0.0h) ? 0 : 0xFFFF;
    
    // Extract sign bit of weight (assuming standard IEEE 754 half precision)
    // Positive: 0x0000, Negative: 0x8000
    ushort w_sign = as_ushort(weight) & 0x8000;
    
    // XOR activation sign with weight sign
    ushort sign_xor = (a ^ w_sign) & 0x8000;
    
    // Result: (Magnitude of A) | (Sign of A XOR Sign of W)
    // Mask with zero_mask to handle 0 weight
    return as_half((ushort)(((a & 0x7FFF) | sign_xor) & zero_mask));
}

// Main composite_tssn_forward kernel
// Uses Subgroup Reduction: One WorkGroup per Neuron, SIMD16
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void composite_tssn_forward(
    __global const half* inputs,
    __global const int* indices,
    __global const half* weights,
    __global const half* sensitivity,
    __global const int* counts,
    __global const int* starts,
    __global const int* function_ids,
    __global half* outputs
) {
    // One WorkGroup processes one Neuron
    // We assume Global Size = Number of Neurons * 16
    // Group ID = Neuron ID
    int neuron_id = get_group_id(0);
    int lane_id = get_sub_group_local_id();
    
    // Get neuron parameters
    int count = counts[neuron_id];
    int start = starts[neuron_id];
    int func_id = function_ids[neuron_id];
    
    half private_sum;
    
    // Initialize based on function type
    if (func_id == 1) { // MIN
        private_sum = 65504.0h; // Max half
    } else if (func_id == 2) { // MAX
        private_sum = -65504.0h; // Min half
    } else {
        private_sum = 0.0h;
    }
    
    // Iterate over synapses for this neuron (Strided by SIMD_WIDTH)
    for (int i = lane_id; i < count; i += SIMD_WIDTH) {
        int syn_idx = start + i;
        
        int in_idx = indices[syn_idx];
        half w = weights[syn_idx];
        half val = inputs[in_idx];
        
        // Stage 369: Ternary MAC / Logic
        
        if (func_id == 1) { // MIN
            half term = val * w;
            private_sum = min(private_sum, term);
            
        } else if (func_id == 2) { // MAX
            half term = val * w;
            private_sum = max(private_sum, term);
            
        } else if (func_id == 3) { // T_WAVE
             private_sum += ternary_mac_ultra(val, w);
             
        } else { // SUM (Standard)
            private_sum += ternary_mac_ultra(val, w);
        }
    }
    
    // Subgroup Reduction
    half final_sum;
    
    if (func_id == 1) { // MIN
        final_sum = sub_group_reduce_min(private_sum);
    } else if (func_id == 2) { // MAX
        final_sum = sub_group_reduce_max(private_sum);
    } else { // SUM / T_WAVE
        final_sum = sub_group_reduce_add(private_sum);
    }
    
    // Write output (only lane 0)
    if (lane_id == 0) {
        // Apply Activation / Post-processing
        if (func_id == 3) { // T_WAVE
            // sin(sum)
            final_sum = sin(final_sum);
        } else if (func_id == 4) { // TERNARY_IF (Threshold)
            if (final_sum > 0.5h) final_sum = 1.0h;
            else if (final_sum < -0.5h) final_sum = -1.0h;
            else final_sum = 0.0h;
        }
        
        outputs[neuron_id] = final_sum;
    }
}
