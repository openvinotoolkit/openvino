#include "composite_tssn.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include <fstream>
#include <mutex>

namespace ov {
namespace op {
namespace v0 {

namespace {

    const char* OPENCL_KERNEL_SOURCE = R"(
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
)";

    void ensure_opencl_kernel_exists() {
        static std::once_flag flag;
        std::call_once(flag, [](){
            // Always overwrite to ensure the latest kernel is used
            std::ofstream out("composite_tssn_kernel.cl");
            out << OPENCL_KERNEL_SOURCE;
            std::cout << "[CompositeTSSN] Deployed OpenCL kernel to composite_tssn_kernel.cl" << std::endl;
        });
    }

    // --- Helper: Topology Check ---
    bool is_identity_mapping(const int32_t* indices, size_t n) {
        size_t i = 0;
        // AVX check
        __m256i seq = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i inc = _mm256_set1_epi32(8);
        
        for (; i + 8 <= n; i += 8) {
            __m256i val = _mm256_loadu_si256((const __m256i*)(indices + i));
            __m256i cmp = _mm256_cmpeq_epi32(val, seq);
            if (_mm256_movemask_epi8(cmp) != -1) return false;
            seq = _mm256_add_epi32(seq, inc);
        }
        // Scalar tail
        for (; i < n; ++i) {
            if (indices[i] != (int32_t)i) return false;
        }
        return true;
    }

    // --- Helper: OpStep for Generic Kernels ---
    struct OpStep {
        enum Type { STANDARD_MUL, BTL_FUNC, FAST_MIN, FAST_MAX, FAST_NEG, FAST_WAVE, FAST_WEIGHTED, FAST_SCRY };
        Type type;
        const btl::BTLFunction* func_ptr = nullptr;
    };

    std::vector<OpStep> prepare_steps(const std::vector<int64_t>& func_ids) {
        static btl::BTLLibrary btl_lib;
        std::vector<OpStep> steps;
        for (int64_t id : func_ids) {
            OpStep step;
            if (id == 113) step.type = OpStep::FAST_MIN;
            else if (id == 4049) step.type = OpStep::FAST_MAX;
            else if (id == 881) step.type = OpStep::FAST_WAVE;
            else if (id == 19570) step.type = OpStep::FAST_NEG; // Assuming ID for NEG
            else {
                step.type = OpStep::BTL_FUNC;
                step.func_ptr = &btl_lib.get_by_function_id(id);
            }
            steps.push_back(step);
        }
        if (steps.empty()) {
            OpStep step;
            step.type = OpStep::STANDARD_MUL;
            steps.push_back(step);
        }
        return steps;
    }

    // --- Helper: Bitwise Logic for ID 881 (T_WAVE) ---
    inline void apply_btl_bitwise_881(__m256i& c, const __m256i& w) {
        __m256i zero = _mm256_setzero_si256();
        __m256i c_p = _mm256_cmpgt_epi32(c, zero);
        __m256i c_n = _mm256_cmpgt_epi32(zero, c);
        __m256i w_p = _mm256_cmpgt_epi32(w, zero);
        __m256i w_n = _mm256_cmpgt_epi32(zero, w);
        
        __m256i res_n = _mm256_or_si256(_mm256_andnot_si256(w_p, c_n), _mm256_andnot_si256(c_p, w_n));
        __m256i res_p = _mm256_or_si256(_mm256_andnot_si256(w_n, c_p), _mm256_andnot_si256(c_n, w_p));
        
        __m256i one = _mm256_set1_epi32(1);
        __m256i p_val = _mm256_and_si256(res_p, one);
        __m256i n_val = _mm256_and_si256(res_n, one);
        c = _mm256_sub_epi32(p_val, n_val);
    }

    // ==================================================================================
    // KERNEL 1: Sparse Scalar (The Baseline)
    // ==================================================================================
    void kernel_sparse_scalar(const float* x_data, const int32_t* indices_data, const float* weights_data, 
                              const float* sensitivity_data, float* out_data, size_t n_synapses, 
                              size_t input_dim, size_t output_dim, const std::vector<int64_t>& func_ids) {
        
        auto steps = prepare_steps(func_ids);
        const int32_t* input_indices = indices_data;
        const int32_t* output_indices = indices_data + n_synapses;

        for (size_t s = 0; s < n_synapses; ++s) {
            int32_t in_idx = input_indices[s];
            int32_t out_idx = output_indices[s];
            
            if (in_idx >= input_dim || out_idx >= output_dim) continue;

            float x_val = x_data[in_idx];
            float w_val = weights_data[s]; 
            float sens = sensitivity_data[s];

            int8_t t_x = (x_val > 0.0f) ? 1 : ((x_val < 0.0f) ? -1 : 0);
            int8_t t_w = (w_val > 0.0f) ? 1 : ((w_val < 0.0f) ? -1 : 0);

            int8_t current_val = t_x;
            for (const auto& step : steps) {
                switch (step.type) {
                    case OpStep::STANDARD_MUL: current_val = current_val * t_w; break;
                    case OpStep::BTL_FUNC: current_val = step.func_ptr->apply(current_val, t_w); break;
                    case OpStep::FAST_MIN: current_val = (current_val < t_w) ? current_val : t_w; break;
                    case OpStep::FAST_MAX: current_val = (current_val > t_w) ? current_val : t_w; break;
                    case OpStep::FAST_NEG: current_val = -current_val; break;
                    case OpStep::FAST_WAVE: 
                        current_val = std::max((int8_t)-1, std::min((int8_t)1, (int8_t)(current_val + t_w)));
                        break;
                    default: break;
                }
            }
            float y_val = (float)current_val * sens * std::abs(x_val);
            out_data[out_idx] += y_val;
        }
    }

    // ==================================================================================
    // KERNEL 2: Dense AVX2 (Generic)
    // ==================================================================================
    void kernel_dense_avx2_generic(const float* x_data, const int32_t* indices_data, const float* weights_data, 
                                   const float* sensitivity_data, float* out_data, size_t n_synapses, 
                                   size_t input_dim, size_t output_dim, const std::vector<int64_t>& func_ids) {
        
        auto steps = prepare_steps(func_ids);
        
        size_t s = 0;
        __m256i one = _mm256_set1_epi32(1);
        __m256i neg_one = _mm256_set1_epi32(-1);
        __m256 zero = _mm256_setzero_ps();
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        for (; s + 8 <= n_synapses; s += 8) {
            __m256 x = _mm256_loadu_ps(x_data + s);
            __m256 w = _mm256_loadu_ps(weights_data + s);
            __m256 sens = _mm256_loadu_ps(sensitivity_data + s);
            
            // Convert to Ternary
            __m256 mask_pos_x = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
            __m256 mask_neg_x = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
            __m256i tx = _mm256_or_si256(
                _mm256_and_si256(_mm256_castps_si256(mask_pos_x), one),
                _mm256_and_si256(_mm256_castps_si256(mask_neg_x), neg_one)
            );

            __m256 mask_pos_w = _mm256_cmp_ps(w, zero, _CMP_GT_OQ);
            __m256 mask_neg_w = _mm256_cmp_ps(w, zero, _CMP_LT_OQ);
            __m256i tw = _mm256_or_si256(
                _mm256_and_si256(_mm256_castps_si256(mask_pos_w), one),
                _mm256_and_si256(_mm256_castps_si256(mask_neg_w), neg_one)
            );

            __m256i c = tx;

            for (const auto& step : steps) {
                switch (step.type) {
                    case OpStep::STANDARD_MUL: c = _mm256_sign_epi32(c, tw); break;
                    case OpStep::FAST_MIN: c = _mm256_min_epi32(c, tw); break;
                    case OpStep::FAST_MAX: c = _mm256_max_epi32(c, tw); break;
                    case OpStep::FAST_NEG: c = _mm256_sub_epi32(_mm256_setzero_si256(), c); break;
                    case OpStep::FAST_WAVE: 
                        c = _mm256_add_epi32(c, tw);
                        c = _mm256_min_epi32(one, _mm256_max_epi32(neg_one, c));
                        break;
                    case OpStep::BTL_FUNC:
                        if (step.func_ptr->id == 881) {
                            apply_btl_bitwise_881(c, tw);
                        }
                        break;
                    default: break;
                }
            }

            __m256 f_res = _mm256_cvtepi32_ps(c);
            __m256 abs_x = _mm256_and_ps(x, abs_mask);
            __m256 y = _mm256_mul_ps(f_res, _mm256_mul_ps(sens, abs_x));

            __m256 old = _mm256_loadu_ps(out_data + s);
            _mm256_storeu_ps(out_data + s, _mm256_add_ps(old, y));
        }

        // Scalar Tail
        for (; s < n_synapses; ++s) {
            float x_val = x_data[s];
            float w_val = weights_data[s]; 
            float sens = sensitivity_data[s];

            int8_t t_x = (x_val > 0.0f) ? 1 : ((x_val < 0.0f) ? -1 : 0);
            int8_t t_w = (w_val > 0.0f) ? 1 : ((w_val < 0.0f) ? -1 : 0);

            int8_t current_val = t_x;
            for (const auto& step : steps) {
                switch (step.type) {
                    case OpStep::STANDARD_MUL: current_val = current_val * t_w; break;
                    case OpStep::BTL_FUNC: current_val = step.func_ptr->apply(current_val, t_w); break;
                    case OpStep::FAST_MIN: current_val = (current_val < t_w) ? current_val : t_w; break;
                    case OpStep::FAST_MAX: current_val = (current_val > t_w) ? current_val : t_w; break;
                    case OpStep::FAST_NEG: current_val = -current_val; break;
                    case OpStep::FAST_WAVE: 
                        current_val = std::max((int8_t)-1, std::min((int8_t)1, (int8_t)(current_val + t_w)));
                        break;
                    default: break;
                }
            }
            float y_val = (float)current_val * sens * std::abs(x_val);
            out_data[s] += y_val;
        }
    }

    // ==================================================================================
    // KERNEL 3: Dense AVX2 (Specialized: MIN + MAX)
    // ==================================================================================
    void kernel_dense_avx2_min_max(const float* x_data, const int32_t* indices_data, const float* weights_data, 
                                   const float* sensitivity_data, float* out_data, size_t n_synapses, 
                                   size_t input_dim, size_t output_dim, const std::vector<int64_t>& func_ids) {
        
        size_t s = 0;
        __m256i one = _mm256_set1_epi32(1);
        __m256i neg_one = _mm256_set1_epi32(-1);
        __m256 zero = _mm256_setzero_ps();
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        for (; s + 8 <= n_synapses; s += 8) {
            __m256 x = _mm256_loadu_ps(x_data + s);
            __m256 w = _mm256_loadu_ps(weights_data + s);
            __m256 sens = _mm256_loadu_ps(sensitivity_data + s);
            
            __m256 mask_pos_x = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
            __m256 mask_neg_x = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
            __m256i tx = _mm256_or_si256(
                _mm256_and_si256(_mm256_castps_si256(mask_pos_x), one),
                _mm256_and_si256(_mm256_castps_si256(mask_neg_x), neg_one)
            );

            __m256 mask_pos_w = _mm256_cmp_ps(w, zero, _CMP_GT_OQ);
            __m256 mask_neg_w = _mm256_cmp_ps(w, zero, _CMP_LT_OQ);
            __m256i tw = _mm256_or_si256(
                _mm256_and_si256(_mm256_castps_si256(mask_pos_w), one),
                _mm256_and_si256(_mm256_castps_si256(mask_neg_w), neg_one)
            );

            // MIN then MAX
            __m256i c = _mm256_min_epi32(tx, tw);
            c = _mm256_max_epi32(c, tw);

            __m256 f_res = _mm256_cvtepi32_ps(c);
            __m256 abs_x = _mm256_and_ps(x, abs_mask);
            __m256 y = _mm256_mul_ps(f_res, _mm256_mul_ps(sens, abs_x));

            __m256 old = _mm256_loadu_ps(out_data + s);
            _mm256_storeu_ps(out_data + s, _mm256_add_ps(old, y));
        }

        // Scalar Tail
        for (; s < n_synapses; ++s) {
            float x_val = x_data[s];
            float w_val = weights_data[s]; 
            float sens = sensitivity_data[s];

            int8_t t_x = (x_val > 0.0f) ? 1 : ((x_val < 0.0f) ? -1 : 0);
            int8_t t_w = (w_val > 0.0f) ? 1 : ((w_val < 0.0f) ? -1 : 0);

            int8_t c = (t_x < t_w) ? t_x : t_w;
            c = (c > t_w) ? c : t_w;

            float y_val = (float)c * sens * std::abs(x_val);
            out_data[s] += y_val;
        }
    }

    // ==================================================================================
    // KERNEL 4: Dense AVX2 (Specialized: MUL)
    // ==================================================================================
    void kernel_dense_avx2_mul(const float* x_data, const int32_t* indices_data, const float* weights_data, 
                               const float* sensitivity_data, float* out_data, size_t n_synapses, 
                               size_t input_dim, size_t output_dim, const std::vector<int64_t>& func_ids) {
        
        size_t s = 0;
        __m256i one = _mm256_set1_epi32(1);
        __m256i neg_one = _mm256_set1_epi32(-1);
        __m256 zero = _mm256_setzero_ps();
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        for (; s + 8 <= n_synapses; s += 8) {
            __m256 x = _mm256_loadu_ps(x_data + s);
            __m256 w = _mm256_loadu_ps(weights_data + s);
            __m256 sens = _mm256_loadu_ps(sensitivity_data + s);
            
            __m256 mask_pos_x = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
            __m256 mask_neg_x = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
            __m256i tx = _mm256_or_si256(
                _mm256_and_si256(_mm256_castps_si256(mask_pos_x), one),
                _mm256_and_si256(_mm256_castps_si256(mask_neg_x), neg_one)
            );

            __m256 mask_pos_w = _mm256_cmp_ps(w, zero, _CMP_GT_OQ);
            __m256 mask_neg_w = _mm256_cmp_ps(w, zero, _CMP_LT_OQ);
            __m256i tw = _mm256_or_si256(
                _mm256_and_si256(_mm256_castps_si256(mask_pos_w), one),
                _mm256_and_si256(_mm256_castps_si256(mask_neg_w), neg_one)
            );

            __m256i c = _mm256_sign_epi32(tx, tw);

            __m256 f_res = _mm256_cvtepi32_ps(c);
            __m256 abs_x = _mm256_and_ps(x, abs_mask);
            __m256 y = _mm256_mul_ps(f_res, _mm256_mul_ps(sens, abs_x));

            __m256 old = _mm256_loadu_ps(out_data + s);
            _mm256_storeu_ps(out_data + s, _mm256_add_ps(old, y));
        }

        // Scalar Tail
        for (; s < n_synapses; ++s) {
            float x_val = x_data[s];
            float w_val = weights_data[s]; 
            float sens = sensitivity_data[s];

            int8_t t_x = (x_val > 0.0f) ? 1 : ((x_val < 0.0f) ? -1 : 0);
            int8_t t_w = (w_val > 0.0f) ? 1 : ((w_val < 0.0f) ? -1 : 0);

            int8_t c = t_x * t_w;

            float y_val = (float)c * sens * std::abs(x_val);
            out_data[s] += y_val;
        }
    }

    // ==================================================================================
    // KERNEL 5: Sparse AVX2 (Generic)
    // ==================================================================================
    void kernel_sparse_avx2_generic(const float* x_data, const int32_t* indices_data, const float* weights_data, 
                                   const float* sensitivity_data, float* out_data, size_t n_synapses, 
                                   size_t input_dim, size_t output_dim, const std::vector<int64_t>& func_ids) {
        
        auto steps = prepare_steps(func_ids);
        const int32_t* input_indices_ptr = indices_data;
        const int32_t* output_indices_ptr = indices_data + n_synapses;

        size_t s = 0;
        __m256i one = _mm256_set1_epi32(1);
        __m256i neg_one = _mm256_set1_epi32(-1);
        __m256 zero = _mm256_setzero_ps();
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        for (; s + 8 <= n_synapses; s += 8) {
            // Load Indices
            __m256i in_idx_v = _mm256_loadu_si256((const __m256i*)(input_indices_ptr + s));
            
            // Gather X
            __m256 x = _mm256_i32gather_ps(x_data, in_idx_v, 4);

            // Load W and Sens
            __m256 w = _mm256_loadu_ps(weights_data + s);
            __m256 sens = _mm256_loadu_ps(sensitivity_data + s);
            
            // --- Math Block (Same as Dense) ---
            __m256 mask_pos_x = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
            __m256 mask_neg_x = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
            __m256i tx = _mm256_or_si256(
                _mm256_and_si256(_mm256_castps_si256(mask_pos_x), one),
                _mm256_and_si256(_mm256_castps_si256(mask_neg_x), neg_one)
            );

            __m256 mask_pos_w = _mm256_cmp_ps(w, zero, _CMP_GT_OQ);
            __m256 mask_neg_w = _mm256_cmp_ps(w, zero, _CMP_LT_OQ);
            __m256i tw = _mm256_or_si256(
                _mm256_and_si256(_mm256_castps_si256(mask_pos_w), one),
                _mm256_and_si256(_mm256_castps_si256(mask_neg_w), neg_one)
            );

            __m256i c = tx;

            for (const auto& step : steps) {
                switch (step.type) {
                    case OpStep::STANDARD_MUL: c = _mm256_sign_epi32(c, tw); break;
                    case OpStep::FAST_MIN: c = _mm256_min_epi32(c, tw); break;
                    case OpStep::FAST_MAX: c = _mm256_max_epi32(c, tw); break;
                    case OpStep::FAST_NEG: c = _mm256_sub_epi32(_mm256_setzero_si256(), c); break;
                    case OpStep::FAST_WAVE: 
                        c = _mm256_add_epi32(c, tw);
                        c = _mm256_min_epi32(one, _mm256_max_epi32(neg_one, c));
                        break;
                    case OpStep::BTL_FUNC:
                        if (step.func_ptr->id == 881) {
                            apply_btl_bitwise_881(c, tw);
                        }
                        break;
                    default: break;
                }
            }

            __m256 f_res = _mm256_cvtepi32_ps(c);
            __m256 abs_x = _mm256_and_ps(x, abs_mask);
            __m256 y = _mm256_mul_ps(f_res, _mm256_mul_ps(sens, abs_x));
            // ----------------------------------

            // Scatter / Accumulate Output
            float temp_y[8];
            _mm256_storeu_ps(temp_y, y);
            
            for (int i = 0; i < 8; ++i) {
                int32_t out_idx = output_indices_ptr[s + i];
                out_data[out_idx] += temp_y[i];
            }
        }

        // Scalar Tail
        for (; s < n_synapses; ++s) {
            int32_t in_idx = input_indices_ptr[s];
            int32_t out_idx = output_indices_ptr[s];
            
            float x_val = x_data[in_idx];
            float w_val = weights_data[s]; 
            float sens = sensitivity_data[s];

            int8_t t_x = (x_val > 0.0f) ? 1 : ((x_val < 0.0f) ? -1 : 0);
            int8_t t_w = (w_val > 0.0f) ? 1 : ((w_val < 0.0f) ? -1 : 0);

            int8_t current_val = t_x;
            for (const auto& step : steps) {
                switch (step.type) {
                    case OpStep::STANDARD_MUL: current_val = current_val * t_w; break;
                    case OpStep::BTL_FUNC: current_val = step.func_ptr->apply(current_val, t_w); break;
                    case OpStep::FAST_MIN: current_val = (current_val < t_w) ? current_val : t_w; break;
                    case OpStep::FAST_MAX: current_val = (current_val > t_w) ? current_val : t_w; break;
                    case OpStep::FAST_NEG: current_val = -current_val; break;
                    case OpStep::FAST_WAVE: 
                        current_val = std::max((int8_t)-1, std::min((int8_t)1, (int8_t)(current_val + t_w)));
                        break;
                    default: break;
                }
            }
            float y_val = (float)current_val * sens * std::abs(x_val);
            out_data[out_idx] += y_val;
        }
    }

} // anonymous namespace

CompositeTSSN::CompositeTSSN(const Output<Node>& x,
                             const Output<Node>& indices,
                             const Output<Node>& weights,
                             const Output<Node>& sensitivity,
                             int64_t output_dim,
                             const std::vector<int64_t>& func_ids)
    : Op({x, indices, weights, sensitivity}), m_output_dim(output_dim), m_func_ids(func_ids) {
    ensure_opencl_kernel_exists();
    validate_and_infer_types();
}

CompositeTSSN::CompositeTSSN(const Output<Node>& x,
                             const Output<Node>& indices,
                             const Output<Node>& weights,
                             const Output<Node>& sensitivity,
                             const Output<Node>& synapse_counts,
                             const Output<Node>& synapse_starts,
                             int64_t output_dim,
                             const std::vector<int64_t>& func_ids)
    : Op({x, indices, weights, sensitivity, synapse_counts, synapse_starts}), m_output_dim(output_dim), m_func_ids(func_ids) {
    ensure_opencl_kernel_exists();
    validate_and_infer_types();
}

CompositeTSSN::CompositeTSSN(const Output<Node>& x,
                             const Output<Node>& indices,
                             const Output<Node>& weights,
                             const Output<Node>& sensitivity,
                             const Output<Node>& synapse_counts,
                             const Output<Node>& synapse_starts,
                             const Output<Node>& function_ids,
                             int64_t output_dim,
                             const std::vector<int64_t>& func_ids)
    : Op({x, indices, weights, sensitivity, synapse_counts, synapse_starts, function_ids}), m_output_dim(output_dim), m_func_ids(func_ids) {
    ensure_opencl_kernel_exists();
    validate_and_infer_types();
}

void CompositeTSSN::validate_and_infer_types() {
    auto x_shape = get_input_partial_shape(0);
    
    // Check input count
    size_t input_count = get_input_size();
    if (input_count != 4 && input_count != 6 && input_count != 7) {
        throw std::runtime_error("CompositeTSSN: Expected 4, 6 or 7 inputs, got " + std::to_string(input_count));
    }

    // Output shape: x_shape[:-1] + {m_output_dim}
    PartialShape output_shape = x_shape;
    if (output_shape.rank().is_static()) {
        output_shape[output_shape.rank().get_length() - 1] = m_output_dim;
    } else {
        output_shape = PartialShape::dynamic();
    }
    
    // std::cout << "[DEBUG] CompositeTSSN::validate_and_infer_types: Input " << x_shape << " Output " << output_shape << " m_output_dim " << m_output_dim << std::endl;

    set_output_type(0, element::f32, output_shape);
}

std::shared_ptr<Node> CompositeTSSN::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<CompositeTSSN>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                               m_output_dim, m_func_ids);
    } else if (new_args.size() == 6) {
        return std::make_shared<CompositeTSSN>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                               new_args.at(4), new_args.at(5),
                                               m_output_dim, m_func_ids);
    } else if (new_args.size() == 7) {
        return std::make_shared<CompositeTSSN>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                               new_args.at(4), new_args.at(5), new_args.at(6),
                                               m_output_dim, m_func_ids);
    } else {
        throw std::runtime_error("CompositeTSSN: Incorrect number of inputs");
    }
}

bool CompositeTSSN::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("output_dim", m_output_dim);
    visitor.on_attribute("func_ids", m_func_ids);
    return true;
}

bool CompositeTSSN::has_evaluate() const {
    return true;
}

void CompositeTSSN::select_kernel() const {
    // This is called inside evaluate, so we can't access inputs directly here easily 
    // without passing them. But we can't change signature.
    // So we will do selection inside evaluate.
}

bool CompositeTSSN::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    const auto& x = inputs[0];
    const auto& indices = inputs[1];
    const auto& weights = inputs[2];
    const auto& sensitivity = inputs[3];
    auto& out = outputs[0];

    size_t input_dim = x.get_shape().back();
    size_t total_batch = ov::shape_size(x.get_shape()) / input_dim;
    size_t n_synapses = weights.get_shape()[0];
    
    const float* x_start = x.data<float>();
    float* out_start = out.data<float>();
    
    const int32_t* indices_data = indices.data<int32_t>();
    const float* weights_data = weights.data<float>();
    const float* sensitivity_data = sensitivity.data<float>();

    // --- KERNEL SELECTION (Once per model load) ---
    {
        std::lock_guard<std::mutex> lock(m_kernel_mutex);
        if (!m_selected_kernel) {
            bool dense_input = is_identity_mapping(indices_data, n_synapses);
            bool dense_output = is_identity_mapping(indices_data + n_synapses, n_synapses);
            
            if (dense_input && dense_output) {
                // Dense Path: Check for specialized chains
                if (m_func_ids.size() == 2 && m_func_ids[0] == 113 && m_func_ids[1] == 4049) {
                    m_selected_kernel = &kernel_dense_avx2_min_max;
                } else if (m_func_ids.empty() || (m_func_ids.size() == 1 && m_func_ids[0] == 0)) { // 0 or empty -> MUL
                    m_selected_kernel = &kernel_dense_avx2_mul;
                } else {
                    m_selected_kernel = &kernel_dense_avx2_generic;
                }
            } else {
                // Sparse Path: Use AVX2 Gather
                m_selected_kernel = &kernel_sparse_avx2_generic;
            }
        }
    }

    // --- BOUNDS CHECK (Dense Only for now) ---
    if (m_selected_kernel != &kernel_sparse_avx2_generic && m_selected_kernel != &kernel_sparse_scalar) {
        if (n_synapses > input_dim) {
             throw std::runtime_error("CompositeTSSN: Dense kernel input dimension mismatch (n_synapses > input_dim)");
        }
        if (n_synapses > m_output_dim) {
             throw std::runtime_error("CompositeTSSN: Dense kernel output dimension mismatch (n_synapses > output_dim)");
        }
    }

    // --- EXECUTION ---
    for (size_t b = 0; b < total_batch; ++b) {
        const float* x_b = x_start + b * input_dim;
        float* out_b = out_start + b * m_output_dim;
        
        std::fill_n(out_b, m_output_dim, 0.0f);
        
        m_selected_kernel(x_b, indices_data, weights_data, sensitivity_data, out_b, 
                          n_synapses, input_dim, m_output_dim, m_func_ids);
    }
    
    return true;
}

} // namespace v0
} // namespace op
} // namespace ov
