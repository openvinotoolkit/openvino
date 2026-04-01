// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <functional>
#include <limits>

#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reduce.hpp>

#include "openvino/core/type/float16.hpp"

#include <cstdlib>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "bench_types.hpp"
#include "bench_attrs.hpp"
#include "bench_utils.hpp"

namespace bench_kernel {

inline bool bench_ref_parallel_enabled() {
    const char* env = std::getenv("OV_BENCH_REF_PARALLEL");
    if (!env || env[0] == '\0')
        return true;

    if (env[0] == '0')
        return false;
    if ((env[0] == 'f' || env[0] == 'F') && (env[1] == 'a' || env[1] == 'A'))
        return false;
    if ((env[0] == 'n' || env[0] == 'N') && (env[1] == 'o' || env[1] == 'O'))
        return false;
    if ((env[0] == 'o' || env[0] == 'O') && (env[1] == 'f' || env[1] == 'F'))
        return false;

    return true;
}

// ============================================================================
// GPU output reader: read memory to std::vector<float>
// ============================================================================

inline std::vector<float> read_memory_to_f32(cldnn::memory::ptr mem, cldnn::stream& stream) {
    // If memory is on device (usm_device), copy to host-accessible memory first
    if (mem->get_allocation_type() == cldnn::allocation_type::usm_device) {
        auto* engine = mem->get_engine();
        auto host_mem = engine->allocate_memory(mem->get_layout(), cldnn::allocation_type::usm_host);
        host_mem->copy_from(stream, *mem, true);
        return read_memory_to_f32(host_mem, stream);
    }

    auto layout = mem->get_layout();
    auto dt = layout.data_type;
    size_t count = layout.count();

    std::vector<float> result(count);

    switch (dt) {
        case cldnn::data_types::f32: {
            auto lock = cldnn::mem_lock<float>(mem, stream);
            for (size_t i = 0; i < count; ++i) result[i] = lock[i];
            break;
        }
        case cldnn::data_types::f16: {
            auto lock = cldnn::mem_lock<ov::float16>(mem, stream);
            for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(lock[i]);
            break;
        }
        case cldnn::data_types::i8: {
            auto lock = cldnn::mem_lock<int8_t>(mem, stream);
            for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(lock[i]);
            break;
        }
        case cldnn::data_types::u8: {
            auto lock = cldnn::mem_lock<uint8_t>(mem, stream);
            for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(lock[i]);
            break;
        }
        case cldnn::data_types::i32: {
            auto lock = cldnn::mem_lock<int32_t>(mem, stream);
            for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(lock[i]);
            break;
        }
        case cldnn::data_types::i64: {
            auto lock = cldnn::mem_lock<int64_t>(mem, stream);
            for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(lock[i]);
            break;
        }
        case cldnn::data_types::i4: {
            auto lock = cldnn::mem_lock<uint8_t>(mem, stream);
            for (size_t i = 0; i < count; ++i) {
                uint8_t byte = lock[i / 2];
                uint8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                int8_t val = (nibble & 0x08) ? static_cast<int8_t>(nibble | 0xF0) : static_cast<int8_t>(nibble);
                result[i] = static_cast<float>(val);
            }
            break;
        }
        case cldnn::data_types::u4: {
            auto lock = cldnn::mem_lock<uint8_t>(mem, stream);
            for (size_t i = 0; i < count; ++i) {
                uint8_t byte = lock[i / 2];
                uint8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                result[i] = static_cast<float>(nibble);
            }
            break;
        }
        default: {
            auto lock = cldnn::mem_lock<uint8_t>(mem, stream);
            size_t byte_count = std::min(count, lock.size());
            for (size_t i = 0; i < byte_count; ++i) result[i] = static_cast<float>(lock[i]);
            break;
        }
    }
    return result;
}

inline std::vector<float> read_network_output_f32(
    const std::map<cldnn::primitive_id, cldnn::network_output>& outputs,
    const std::string& prim_id,
    cldnn::stream& stream) {
    auto it = outputs.find(prim_id);
    if (it == outputs.end()) return {};
    auto mem = it->second.get_memory();
    if (!mem) return {};
    return read_memory_to_f32(mem, stream);
}

// ============================================================================
// Accuracy comparison
// ============================================================================

struct acc_result {
    bool pass = false;
    size_t total_elements = 0;
    size_t mismatches = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    size_t max_abs_idx = 0;
    float max_abs_ref = 0.0f;
    float max_abs_gpu = 0.0f;

    void print(const std::string& label = "", int verbose = 1) const {
        if (label.empty())
            std::cout << "acc: ";
        else
            std::cout << label << " acc: ";

        if (pass) {
            std::cout << "PASS";
        } else {
            std::cout << "FAIL";
        }

        std::cout << " (elements=" << total_elements
                  << " mismatches=" << mismatches
                  << " max_abs=" << std::scientific << std::setprecision(4) << max_abs_diff
                  << " max_rel=" << max_rel_diff;
        if (verbose >= 2 && mismatches > 0) {
            std::cout << " worst_idx=" << max_abs_idx
                      << " ref=" << std::fixed << std::setprecision(6) << max_abs_ref
                      << " gpu=" << max_abs_gpu;
        }
        std::cout << ")" << std::endl;
    }
};

// Compare GPU output vs CPU reference with tolerance
// atol/rtol: absolute/relative tolerance
// threshold: max fraction of elements allowed to fail
inline acc_result compare_f32(const std::vector<float>& gpu,
                               const std::vector<float>& ref,
                               float atol = 1e-3f,
                               float rtol = 1e-2f,
                               float threshold = 0.0f) {
    acc_result res;
    res.total_elements = ref.size();

    // Guard: empty vectors are trivially equal
    if (gpu.empty() && ref.empty()) {
        res.pass = true;
        return res;
    }

    if (gpu.size() != ref.size()) {
        res.pass = false;
        res.mismatches = std::max(gpu.size(), ref.size());
        return res;
    }

    for (size_t i = 0; i < ref.size(); ++i) {
        float r = ref[i];
        float g = gpu[i];

        // Handle NaN
        if (std::isnan(r) && std::isnan(g)) continue;
        if (std::isnan(r) || std::isnan(g)) {
            res.mismatches++;
            continue;
        }

        // Handle Inf: same-sign inf is a match; GPU inf with large ref is f16 overflow
        if (std::isinf(g) || std::isinf(r)) {
            if (std::isinf(g) && std::isinf(r) && ((g > 0) == (r > 0))) continue;  // same-sign inf
            constexpr float f16_max = 65504.0f;
            if (std::isinf(g) && g > 0 && r > f16_max) continue;   // f16 overflow to +inf
            if (std::isinf(g) && g < 0 && r < -f16_max) continue;  // f16 overflow to -inf
            res.mismatches++;
            continue;
        }

        float abs_diff = std::fabs(g - r);
        float rel_diff = (std::fabs(r) > 1e-8f) ? abs_diff / std::fabs(r) : abs_diff;

        bool ok = (abs_diff <= atol) || (rel_diff <= rtol);
        if (!ok) {
            res.mismatches++;
        }

        if (abs_diff > res.max_abs_diff) {
            res.max_abs_diff = abs_diff;
            res.max_abs_idx = i;
            res.max_abs_ref = r;
            res.max_abs_gpu = g;
        }
        if (rel_diff > res.max_rel_diff) {
            res.max_rel_diff = rel_diff;
        }
    }

    float mismatch_ratio = static_cast<float>(res.mismatches) / static_cast<float>(res.total_elements);
    res.pass = (mismatch_ratio <= threshold);
    return res;
}

// ============================================================================
// Default tolerance per data type
// ============================================================================

inline void get_default_tolerance(cldnn::data_types dt, float& atol, float& rtol) {
    switch (dt) {
        case cldnn::data_types::f32:
            atol = 1e-5f; rtol = 1e-4f; break;
        case cldnn::data_types::f16:
            atol = 5e-3f; rtol = 5e-2f; break;
        case cldnn::data_types::i8:
        case cldnn::data_types::u8:
            atol = 1.0f; rtol = 0.1f; break;
        default:
            atol = 1e-2f; rtol = 5e-2f; break;
    }
}

// ============================================================================
// CPU reference implementations for each kernel type
//
// All operate on flattened float vectors.
// Shapes are provided as std::vector<int64_t>.
// ============================================================================

namespace ref {

// --- Helper: flat index calculation ---
inline size_t flat_index(const std::vector<int64_t>& shape, const std::vector<size_t>& indices) {
    size_t idx = 0;
    size_t stride = 1;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        idx += indices[d] * stride;
        stride *= static_cast<size_t>(shape[d]);
    }
    return idx;
}

inline size_t total_elements(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
}

// --- Fully Connected: output[b][o] = sum_i(input[b][i] * weight[o][i]) ---
// input: [B, IC], weight: [OC, IC], output: [B, OC]
inline std::vector<float> fc(const std::vector<float>& input,
                              const std::vector<int64_t>& input_shape,
                              const std::vector<float>& weight,
                              const std::vector<int64_t>& weight_shape) {
    int64_t IC = input_shape.back();
    // For 2D weight [OC, IC] or 3D weight [batch, OC, IC], OC is always at dim [-2]
    int64_t OC = weight_shape.size() >= 2 ? weight_shape[weight_shape.size() - 2] : weight_shape[0];

    // Check for 3D batched FC: input [batch, seq, IC], weight [batch, OC, IC]
    // Each batch uses its own weight matrix
    bool batched_weights = (weight_shape.size() >= 3 && input_shape.size() >= 3 &&
                            weight_shape[0] == input_shape[0]);

    if (batched_weights) {
        int64_t batch = weight_shape[0];
        int64_t seq = 1;
        for (size_t i = 1; i < input_shape.size() - 1; ++i) seq *= input_shape[i];

        std::vector<float> output(static_cast<size_t>(batch * seq * OC), 0.0f);

        if (bench_ref_parallel_enabled() && batch >= 4) {
            tbb::parallel_for(tbb::blocked_range<int64_t>(0, batch), [&](const tbb::blocked_range<int64_t>& r) {
                for (int64_t n = r.begin(); n != r.end(); ++n) {
                    for (int64_t s = 0; s < seq; ++s) {
                        for (int64_t o = 0; o < OC; ++o) {
                            float sum = 0.0f;
                            for (int64_t i = 0; i < IC; ++i) {
                                sum += input[(n * seq + s) * IC + i] * weight[(n * OC + o) * IC + i];
                            }
                            output[(n * seq + s) * OC + o] = sum;
                        }
                    }
                }
            });
        } else {
            for (int64_t n = 0; n < batch; ++n) {
                for (int64_t s = 0; s < seq; ++s) {
                    for (int64_t o = 0; o < OC; ++o) {
                        float sum = 0.0f;
                        for (int64_t i = 0; i < IC; ++i) {
                            sum += input[(n * seq + s) * IC + i] * weight[(n * OC + o) * IC + i];
                        }
                        output[(n * seq + s) * OC + o] = sum;
                    }
                }
            }
        }
        return output;
    }

    // Standard (non-batched) FC
    int64_t B = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) B *= input_shape[i];

    std::vector<float> output(static_cast<size_t>(B * OC), 0.0f);

    if (bench_ref_parallel_enabled() && (B * OC) >= 1024) {
        // Parallelize over B×OC flat space to cover both large-batch and large-OC cases
        // (e.g., lm_head: B=1, OC=128256 would not fire under the old B>=4 condition)
        tbb::parallel_for(tbb::blocked_range<int64_t>(0, B * OC), [&](const tbb::blocked_range<int64_t>& r) {
            for (int64_t bo = r.begin(); bo != r.end(); ++bo) {
                int64_t b = bo / OC;
                int64_t o = bo % OC;
                float sum = 0.0f;
                for (int64_t i = 0; i < IC; ++i) {
                    sum += input[b * IC + i] * weight[o * IC + i];
                }
                output[bo] = sum;
            }
        });
    } else {
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t o = 0; o < OC; ++o) {
                float sum = 0.0f;
                for (int64_t i = 0; i < IC; ++i) {
                    sum += input[b * IC + i] * weight[o * IC + i];
                }
                output[b * OC + o] = sum;
            }
        }
    }
    return output;
}

// --- Weight decompression for compressed FC (i4/u4/i8/u8 weights) ---
// raw_weights: quantized weight values read as float
// weight_shape: [OC, IC]
// scales: decompression scale, scale_shape: [OC, n_groups] or [OC, 1]
// zero_points: optional, same shape as scales
inline std::vector<float> decompress_weights(
    const std::vector<float>& raw_weights,
    const std::vector<int64_t>& weight_shape,
    const std::vector<float>& scales,
    const std::vector<int64_t>& scale_shape,
    const std::vector<float>& zero_points = {},
    const std::vector<int64_t>& zp_shape = {}) {
    (void)zp_shape;  // same as scale_shape
    int64_t OC = weight_shape[0];
    int64_t IC = weight_shape[1];
    int64_t n_groups = scale_shape.size() > 1 ? scale_shape[1] : 1;
    int64_t group_size = (n_groups > 1) ? ((IC + n_groups - 1) / n_groups) : IC;

    std::vector<float> result(raw_weights.size());
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t ic = 0; ic < IC; ++ic) {
            size_t idx = static_cast<size_t>(oc * IC + ic);
            int64_t g = std::min(ic / group_size, n_groups - 1);
            size_t sg_idx = static_cast<size_t>(oc * n_groups + g);

            float w = raw_weights[idx];
            float s = scales[sg_idx];
            float zp = zero_points.empty() ? 0.0f : zero_points[sg_idx];
            result[idx] = (w - zp) * s;
        }
    }
    return result;
}

// --- Tensor permutation (like numpy.transpose) ---
// Reorders tensor data according to the given order.
// E.g., shape [1,128,12,64] with order [0,2,1,3] → shape [1,12,128,64]
inline void permute_tensor(const std::vector<float>& input,
                           const std::vector<int64_t>& input_shape,
                           const std::vector<int64_t>& order,
                           std::vector<float>& output,
                           std::vector<int64_t>& output_shape) {
    size_t rank = input_shape.size();

    // Compute output shape
    output_shape.resize(rank);
    for (size_t i = 0; i < rank; i++)
        output_shape[i] = input_shape[order[i]];

    // Compute input strides (row-major)
    std::vector<int64_t> in_strides(rank);
    in_strides[rank - 1] = 1;
    for (int i = static_cast<int>(rank) - 2; i >= 0; i--)
        in_strides[i] = in_strides[i + 1] * input_shape[i + 1];

    // Compute output strides
    std::vector<int64_t> out_strides(rank);
    out_strides[rank - 1] = 1;
    for (int i = static_cast<int>(rank) - 2; i >= 0; i--)
        out_strides[i] = out_strides[i + 1] * output_shape[i + 1];

    size_t total = input.size();
    output.resize(total);

    // For each output position, find corresponding input position
    for (size_t flat = 0; flat < total; flat++) {
        // Decompose flat into output multi-dim indices
        int64_t in_flat = 0;
        int64_t remaining = static_cast<int64_t>(flat);
        for (size_t d = 0; d < rank; d++) {
            int64_t coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            // output[d] came from input[order[d]], so input index for this dim is coord
            in_flat += coord * in_strides[order[d]];
        }
        output[flat] = input[in_flat];
    }
}

// Check if order is identity permutation [0, 1, 2, ...]
inline bool is_identity_order(const std::vector<int64_t>& order) {
    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] != static_cast<int64_t>(i)) return false;
    }
    return true;
}

// --- Gemm: C = A * B (with optional transpose) ---
// transpose_a=false: A is [..., M, K];  transpose_a=true: A is [..., K, M]
// transpose_b=false: B is [..., K, N];  transpose_b=true: B is [..., N, K]
// Output: [..., M, N]
inline std::vector<float> gemm(const std::vector<float>& A,
                                const std::vector<int64_t>& A_shape,
                                const std::vector<float>& B,
                                const std::vector<int64_t>& B_shape,
                                bool transpose_a = false,
                                bool transpose_b = false) {
    size_t rank = A_shape.size();

    // Determine M, K, N based on transpose flags
    int64_t M, K, K2, N;
    if (transpose_a) {
        // A is [..., K, M]
        K = A_shape[rank - 2];
        M = A_shape[rank - 1];
    } else {
        // A is [..., M, K]
        M = A_shape[rank - 2];
        K = A_shape[rank - 1];
    }
    if (transpose_b) {
        // B is [..., N, K]
        N = B_shape[rank - 2];
        K2 = B_shape[rank - 1];
    } else {
        // B is [..., K, N]
        K2 = B_shape[rank - 2];
        N = B_shape[rank - 1];
    }
    (void)K2;  // K should equal K2

    // A physical layout strides (innermost 2 dims)
    int64_t A_row_stride = A_shape[rank - 1];  // physical row stride
    int64_t A_col_stride = 1;
    // B physical layout strides
    int64_t B_row_stride = B_shape[rank - 1];
    int64_t B_col_stride = 1;

    // Batch size = product of all dims except last 2
    int64_t batch = 1;
    for (size_t i = 0; i < rank - 2; ++i) batch *= A_shape[i];

    int64_t A_batch_stride = A_shape[rank - 2] * A_shape[rank - 1];
    int64_t B_batch_stride = B_shape[rank - 2] * B_shape[rank - 1];

    std::vector<float> C(static_cast<size_t>(batch * M * N), 0.0f);

    for (int64_t b = 0; b < batch; ++b) {
        const float* Ap = A.data() + b * A_batch_stride;
        const float* Bp = B.data() + b * B_batch_stride;
        float* Cp = C.data() + b * M * N;

        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    // A[m,k]: if transpose_a, physical is [k,m], else [m,k]
                    float a_val = transpose_a
                        ? Ap[k * A_row_stride + m * A_col_stride]
                        : Ap[m * A_row_stride + k * A_col_stride];
                    // B[k,n]: if transpose_b, physical is [n,k], else [k,n]
                    float b_val = transpose_b
                        ? Bp[n * B_row_stride + k * B_col_stride]
                        : Bp[k * B_row_stride + n * B_col_stride];
                    sum += a_val * b_val;
                }
                Cp[m * N + n] = sum;
            }
        }
    }
    return C;
}

// --- Convolution: 2D, no bias, groups=1, stride=1, dilation=1, pad=0 ---
// input: [N, C, H, W], weight: [OC, C, KH, KW], output: [N, OC, OH, OW]
inline std::vector<float> conv2d(const std::vector<float>& input,
                                  const std::vector<int64_t>& input_shape,
                                  const std::vector<float>& weight,
                                  const std::vector<int64_t>& weight_shape) {
    int64_t N  = input_shape[0], IC = input_shape[1], IH = input_shape[2], IW = input_shape[3];
    int64_t OC = weight_shape[0], KH = weight_shape[2], KW = weight_shape[3];
    int64_t OH = IH - KH + 1;
    int64_t OW = IW - KW + 1;

    std::vector<float> output(static_cast<size_t>(N * OC * OH * OW), 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int64_t ic = 0; ic < IC; ++ic) {
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                int64_t ih = oh + kh;
                                int64_t iw = ow + kw;
                                sum += input[((n * IC + ic) * IH + ih) * IW + iw]
                                     * weight[((oc * IC + ic) * KH + kh) * KW + kw];
                            }
                        }
                    }
                    output[((n * OC + oc) * OH + oh) * OW + ow] = sum;
                }
            }
        }
    }
    return output;
}

// --- Generic N-D Convolution with stride, dilation, padding ---
// input: [N, C, spatial...], weight: [OC, C/groups, spatial_k...]
inline std::vector<float> convNd(const std::vector<float>& input,
                                 const std::vector<int64_t>& input_shape,
                                 const std::vector<float>& weight,
                                 const std::vector<int64_t>& weight_shape,
                                 int64_t groups = 1,
                                 const std::vector<size_t>& strides_in = {},
                                 const std::vector<size_t>& dilations_in = {},
                                 const std::vector<ptrdiff_t>& padding_begin_in = {},
                                 const std::vector<ptrdiff_t>& padding_end_in = {}) {
    size_t rank = input_shape.size();
    size_t spatial_dims = rank - 2;
    int64_t N = input_shape[0];
    int64_t IC = input_shape[1];
    int64_t OC = weight_shape[0];
    int64_t IC_per_group = weight_shape[1];
    int64_t OC_per_group = OC / groups;

    // Default stride=1, dilation=1, pad=0
    std::vector<int64_t> strides(spatial_dims, 1);
    std::vector<int64_t> dilations(spatial_dims, 1);
    std::vector<int64_t> pad_begin(spatial_dims, 0);
    std::vector<int64_t> pad_end(spatial_dims, 0);
    for (size_t d = 0; d < spatial_dims; ++d) {
        if (d < strides_in.size()) strides[d] = static_cast<int64_t>(strides_in[d]);
        if (d < dilations_in.size()) dilations[d] = static_cast<int64_t>(dilations_in[d]);
        if (d < padding_begin_in.size()) pad_begin[d] = static_cast<int64_t>(padding_begin_in[d]);
        if (d < padding_end_in.size()) pad_end[d] = static_cast<int64_t>(padding_end_in[d]);
    }

    std::vector<int64_t> out_spatial(spatial_dims);
    std::vector<int64_t> kernel_dims(spatial_dims);
    int64_t out_spatial_total = 1;
    int64_t kernel_total = 1;
    for (size_t d = 0; d < spatial_dims; ++d) {
        kernel_dims[d] = weight_shape[d + 2];
        // OH = (IH + pad_begin + pad_end - dilation*(KH-1) - 1) / stride + 1
        int64_t effective_k = dilations[d] * (kernel_dims[d] - 1) + 1;
        out_spatial[d] = (input_shape[d + 2] + pad_begin[d] + pad_end[d] - effective_k) / strides[d] + 1;
        out_spatial_total *= out_spatial[d];
        kernel_total *= kernel_dims[d];
    }

    size_t out_total = static_cast<size_t>(N * OC * out_spatial_total);
    std::vector<float> output(out_total, 0.0f);

    // ---- Fast path for 2D convolution (most common case) ----
    if (spatial_dims == 2) {
        int64_t IH = input_shape[2], IW = input_shape[3];
        int64_t KH = kernel_dims[0], KW = kernel_dims[1];
        int64_t OH = out_spatial[0], OW = out_spatial[1];
        int64_t sH = strides[0], sW = strides[1];
        int64_t dH = dilations[0], dW = dilations[1];
        int64_t pH = pad_begin[0], pW = pad_begin[1];

        auto conv2d_kernel = [&](int64_t oc_begin, int64_t oc_end) {
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t oc = oc_begin; oc < oc_end; ++oc) {
                    int64_t g = oc / OC_per_group;
                    int64_t ic_start = g * IC_per_group;
                    for (int64_t oh = 0; oh < OH; ++oh) {
                        for (int64_t ow = 0; ow < OW; ++ow) {
                            float sum = 0.0f;
                            for (int64_t ic = 0; ic < IC_per_group; ++ic) {
                                for (int64_t kh = 0; kh < KH; ++kh) {
                                    int64_t ih = oh * sH - pH + kh * dH;
                                    if (ih < 0 || ih >= IH) continue;
                                    for (int64_t kw = 0; kw < KW; ++kw) {
                                        int64_t iw = ow * sW - pW + kw * dW;
                                        if (iw < 0 || iw >= IW) continue;
                                        sum += input[((n * IC + ic_start + ic) * IH + ih) * IW + iw]
                                             * weight[(oc * IC_per_group + ic) * KH * KW + kh * KW + kw];
                                    }
                                }
                            }
                            output[((n * OC + oc) * OH + oh) * OW + ow] = sum;
                        }
                    }
                }
            }
        };

        // The conv2d_kernel lambda already loops over n=0..N internally,
        // so we call it once (not inside an outer N loop).
        if (bench_ref_parallel_enabled() && OC >= 4) {
            tbb::parallel_for(tbb::blocked_range<int64_t>(0, OC), [&](const tbb::blocked_range<int64_t>& r) {
                conv2d_kernel(r.begin(), r.end());
            });
        } else {
            conv2d_kernel(0, OC);
        }
        return output;
    }

    // ---- Fast path for 1D convolution ----
    if (spatial_dims == 1) {
        int64_t IL = input_shape[2];
        int64_t KL = kernel_dims[0];
        int64_t OL = out_spatial[0];
        int64_t sL = strides[0], dL = dilations[0], pL = pad_begin[0];

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t oc = 0; oc < OC; ++oc) {
                int64_t g = oc / OC_per_group;
                int64_t ic_start = g * IC_per_group;
                for (int64_t ol = 0; ol < OL; ++ol) {
                    float sum = 0.0f;
                    for (int64_t ic = 0; ic < IC_per_group; ++ic) {
                        for (int64_t k = 0; k < KL; ++k) {
                            int64_t il = ol * sL - pL + k * dL;
                            if (il < 0 || il >= IL) continue;
                            sum += input[(n * IC + ic_start + ic) * IL + il]
                                 * weight[(oc * IC_per_group + ic) * KL + k];
                        }
                    }
                    output[(n * OC + oc) * OL + ol] = sum;
                }
            }
        }
        return output;
    }

    // ---- Generic N-D fallback (with multi-threading over OC) ----
    auto convNd_kernel = [&](int64_t oc_begin, int64_t oc_end) {
        std::vector<int64_t> out_coords(spatial_dims);
        std::vector<int64_t> k_coords(spatial_dims);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t oc = oc_begin; oc < oc_end; ++oc) {
                int64_t g = oc / OC_per_group;
                int64_t ic_start = g * IC_per_group;

                for (int64_t os = 0; os < out_spatial_total; ++os) {
                    {
                        int64_t rem = os;
                        for (int sd = static_cast<int>(spatial_dims) - 1; sd >= 0; --sd) {
                            out_coords[sd] = rem % out_spatial[sd];
                            rem /= out_spatial[sd];
                        }
                    }

                    float sum = 0.0f;
                    for (int64_t ic = 0; ic < IC_per_group; ++ic) {
                        for (int64_t ki = 0; ki < kernel_total; ++ki) {
                            {
                                int64_t rem = ki;
                                for (int sd = static_cast<int>(spatial_dims) - 1; sd >= 0; --sd) {
                                    k_coords[sd] = rem % kernel_dims[sd];
                                    rem /= kernel_dims[sd];
                                }
                            }

                            // Check padding bounds for each spatial dim
                            bool in_bounds = true;
                            size_t in_idx = static_cast<size_t>(n);
                            in_idx = in_idx * static_cast<size_t>(IC) + static_cast<size_t>(ic_start + ic);
                            for (size_t d = 0; d < spatial_dims; ++d) {
                                int64_t pos = out_coords[d] * strides[d] - pad_begin[d] + k_coords[d] * dilations[d];
                                if (pos < 0 || pos >= input_shape[d + 2]) { in_bounds = false; break; }
                                in_idx = in_idx * static_cast<size_t>(input_shape[d + 2])
                                       + static_cast<size_t>(pos);
                            }
                            if (!in_bounds) continue;

                            size_t w_idx = static_cast<size_t>(oc);
                            w_idx = w_idx * static_cast<size_t>(IC_per_group) + static_cast<size_t>(ic);
                            for (size_t d = 0; d < spatial_dims; ++d) {
                                w_idx = w_idx * static_cast<size_t>(kernel_dims[d])
                                      + static_cast<size_t>(k_coords[d]);
                            }

                            sum += input[in_idx] * weight[w_idx];
                        }
                    }

                    size_t out_idx = static_cast<size_t>(n);
                    out_idx = out_idx * static_cast<size_t>(OC) + static_cast<size_t>(oc);
                    for (size_t d = 0; d < spatial_dims; ++d) {
                        out_idx = out_idx * static_cast<size_t>(out_spatial[d])
                                + static_cast<size_t>(out_coords[d]);
                    }
                    output[out_idx] = sum;
                }
            }
        }
    };

    {
        if (bench_ref_parallel_enabled() && OC >= 4) {
            tbb::parallel_for(tbb::blocked_range<int64_t>(0, OC), [&](const tbb::blocked_range<int64_t>& r) {
                convNd_kernel(r.begin(), r.end());
            });
        } else {
            convNd_kernel(0, OC);
        }
    }
    return output;
}

// --- Softmax along arbitrary axis (default: last dimension) ---
inline std::vector<float> softmax(const std::vector<float>& input,
                                   const std::vector<int64_t>& shape,
                                   int64_t axis = -1) {
    size_t rank = shape.size();
    if (axis < 0) axis += static_cast<int64_t>(rank);
    size_t total = total_elements(shape);
    int64_t dim_size = shape[axis];

    // Compute outer (before axis) and inner (after axis) strides
    size_t outer = 1;
    for (int64_t d = 0; d < axis; ++d) outer *= static_cast<size_t>(shape[d]);
    size_t inner = 1;
    for (size_t d = static_cast<size_t>(axis) + 1; d < rank; ++d) inner *= static_cast<size_t>(shape[d]);

    std::vector<float> output(total);

    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            // Find max for numeric stability
            float max_val = -1e30f;
            for (int64_t a = 0; a < dim_size; ++a) {
                size_t idx = (o * static_cast<size_t>(dim_size) + static_cast<size_t>(a)) * inner + i;
                max_val = std::max(max_val, input[idx]);
            }
            float sum_exp = 0.0f;
            for (int64_t a = 0; a < dim_size; ++a) {
                size_t idx = (o * static_cast<size_t>(dim_size) + static_cast<size_t>(a)) * inner + i;
                output[idx] = std::exp(input[idx] - max_val);
                sum_exp += output[idx];
            }
            for (int64_t a = 0; a < dim_size; ++a) {
                size_t idx = (o * static_cast<size_t>(dim_size) + static_cast<size_t>(a)) * inner + i;
                output[idx] /= sum_exp;
            }
        }
    }
    return output;
}

// --- Activation functions ---
inline float apply_activation(float x, cldnn::activation_func func, float alpha = 0.0f, float beta = 0.0f) {
    switch (func) {
        case cldnn::activation_func::relu:
            return x > 0.0f ? x : 0.0f;
        case cldnn::activation_func::relu_negative_slope:
            return x > 0.0f ? x : alpha * x;
        case cldnn::activation_func::logistic:  // sigmoid
            return 1.0f / (1.0f + std::exp(-x));
        case cldnn::activation_func::hyperbolic_tan:
            return std::tanh(x);
        case cldnn::activation_func::elu:
            return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f);
        case cldnn::activation_func::abs:
            return std::fabs(x);
        case cldnn::activation_func::sqrt:
            return std::sqrt(std::fabs(x));
        case cldnn::activation_func::square:
            return x * x;
        case cldnn::activation_func::exp:
            return std::exp(x);
        case cldnn::activation_func::log:
            return std::log(x);
        case cldnn::activation_func::gelu: {
            // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
            return x * 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
        }
        case cldnn::activation_func::gelu_tanh: {
            // GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float k = std::sqrt(2.0f / 3.14159265358979f);
            return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
        }
        case cldnn::activation_func::swish:
            // swish(x) = x * sigmoid(alpha * x); alpha defaults to 1.0 when not specified
            return x / (1.0f + std::exp(-(alpha != 0.0f ? alpha : 1.0f) * x));
        case cldnn::activation_func::hswish: {
            float t = std::min(std::max(x + 3.0f, 0.0f), 6.0f);
            return x * t / 6.0f;
        }
        case cldnn::activation_func::mish:
            return x * std::tanh(std::log(1.0f + std::exp(x)));
        case cldnn::activation_func::hard_sigmoid: {
            return std::min(std::max(alpha * x + beta, 0.0f), 1.0f);
        }
        case cldnn::activation_func::hsigmoid: {
            float t = std::min(std::max(x + 3.0f, 0.0f), 6.0f);
            return t / 6.0f;
        }
        case cldnn::activation_func::clamp:
            return std::min(std::max(x, alpha), beta);
        case cldnn::activation_func::negative:
            return -x;
        case cldnn::activation_func::softplus:
            return std::log(1.0f + std::exp(x));
        case cldnn::activation_func::softsign:
            return x / (1.0f + std::fabs(x));
        case cldnn::activation_func::linear:
            return alpha * x + beta;
        case cldnn::activation_func::round_half_to_even:
            return std::nearbyint(x);
        default:
            return x;
    }
}

inline std::vector<float> activation(const std::vector<float>& input,
                                      cldnn::activation_func func,
                                      float alpha = 0.0f,
                                      float beta = 0.0f) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = apply_activation(input[i], func, alpha, beta);
    }
    return output;
}

// --- Apply post-ops to reference output ---
// Supports activation and eltwise post-ops.
// elt_data_map: post-op index -> {data, shape} for eltwise post-ops
struct elt_ref_data {
    std::vector<float> data;
    std::vector<int64_t> shape;  // for broadcasting
};

inline void apply_eltwise_broadcast(std::vector<float>& output,
                                    const std::vector<int64_t>& out_shape,
                                    const std::vector<float>& elt_data,
                                    const std::vector<int64_t>& elt_shape,
                                    eltwise_mode mode) {
    size_t rank = out_shape.size();
    size_t total = total_elements(out_shape);

    // Pad elt_shape to same rank (prepend 1s if needed)
    std::vector<int64_t> elt_padded(rank, 1);
    size_t elt_rank = elt_shape.size();
    size_t pad_offset = (elt_rank <= rank) ? (rank - elt_rank) : 0;
    for (size_t d = 0; d < std::min(elt_rank, rank); ++d) {
        elt_padded[pad_offset + d] = elt_shape[d];
    }

    // Compute strides
    std::vector<size_t> out_strides(rank, 1);
    std::vector<size_t> elt_strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
        out_strides[d] = out_strides[d + 1] * static_cast<size_t>(out_shape[d + 1]);
        elt_strides[d] = elt_strides[d + 1] * static_cast<size_t>(elt_padded[d + 1]);
    }

    for (size_t idx = 0; idx < total; ++idx) {
        size_t remaining = idx;
        size_t elt_idx = 0;
        for (size_t d = 0; d < rank; ++d) {
            size_t coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            size_t ec = (elt_padded[d] > 1) ? coord : 0;
            elt_idx += ec * elt_strides[d];
        }

        float ev = elt_data[elt_idx];
        switch (mode) {
            case eltwise_mode::sum:      output[idx] += ev; break;
            case eltwise_mode::prod:     output[idx] *= ev; break;
            case eltwise_mode::sub:      output[idx] -= ev; break;
            case eltwise_mode::div_mode: output[idx] /= ev; break;
        }
    }
}

// --- Quantize reference: clamp + round to target data type ---
inline void apply_quantize_ref(std::vector<float>& data, cldnn::data_types out_dt) {
    float lo = 0, hi = 0;
    bool need_round = false;
    switch (out_dt) {
        case cldnn::data_types::u8:  lo = 0;    hi = 255;  need_round = true; break;
        case cldnn::data_types::i8:  lo = -128;  hi = 127;  need_round = true; break;
        case cldnn::data_types::u4:  lo = 0;    hi = 15;   need_round = true; break;
        case cldnn::data_types::i4:  lo = -8;   hi = 7;    need_round = true; break;
        case cldnn::data_types::i32: lo = -2147483648.0f; hi = 2147483647.0f; need_round = true; break;
        default: return;  // f32/f16 - no clamp/round needed
    }
    for (auto& v : data) {
        v = std::min(std::max(v, lo), hi);
        if (need_round) v = std::nearbyint(v);
    }
}

inline bool apply_post_ops_ref(std::vector<float>& data,
                               const std::vector<post_op_entry>& post_ops,
                               const std::vector<int64_t>& output_shape = {},
                               const std::map<int, elt_ref_data>& elt_data_map = {}) {
    for (size_t pi = 0; pi < post_ops.size(); ++pi) {
        const auto& po = post_ops[pi];
        if (po.kind == post_op_kind::activation) {
            auto func = map_activation(po.act_func, po.alpha);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = apply_activation(data[i], func, po.alpha, po.beta);
            }
        } else if (po.kind == post_op_kind::eltwise) {
            auto it = elt_data_map.find(static_cast<int>(pi));
            if (it == elt_data_map.end()) return false;  // no data provided
            apply_eltwise_broadcast(data, output_shape,
                                    it->second.data, it->second.shape, po.elt_mode);
        } else if (po.kind == post_op_kind::quantize) {
            apply_quantize_ref(data, po.quant_out_dt);
        } else {
            // swiglu - not supported in bench topology
            return false;
        }
    }
    return true;
}

// --- Eltwise with numpy-style broadcasting ---
inline std::vector<float> eltwise(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   cldnn::eltwise_mode mode,
                                   const std::vector<int64_t>& a_shape = {},
                                   const std::vector<int64_t>& b_shape = {}) {
    // If shapes are provided and differ, use numpy-style broadcasting
    if (!a_shape.empty() && !b_shape.empty() && a_shape != b_shape) {
        // Pad shapes to same rank
        size_t rank = std::max(a_shape.size(), b_shape.size());
        std::vector<int64_t> sa(rank, 1), sb(rank, 1), so(rank);
        for (size_t i = 0; i < a_shape.size(); ++i)
            sa[rank - a_shape.size() + i] = a_shape[i];
        for (size_t i = 0; i < b_shape.size(); ++i)
            sb[rank - b_shape.size() + i] = b_shape[i];
        size_t out_total = 1;
        for (size_t i = 0; i < rank; ++i) {
            so[i] = std::max(sa[i], sb[i]);
            out_total *= static_cast<size_t>(so[i]);
        }

        // Compute strides for a, b in the output index space
        std::vector<size_t> a_strides(rank), b_strides(rank), o_strides(rank);
        a_strides[rank-1] = 1; b_strides[rank-1] = 1; o_strides[rank-1] = 1;
        for (int d = static_cast<int>(rank)-2; d >= 0; --d) {
            a_strides[d] = a_strides[d+1] * static_cast<size_t>(sa[d+1]);
            b_strides[d] = b_strides[d+1] * static_cast<size_t>(sb[d+1]);
            o_strides[d] = o_strides[d+1] * static_cast<size_t>(so[d+1]);
        }

        std::vector<float> output(out_total);
        for (size_t idx = 0; idx < out_total; ++idx) {
            // Decompose flat index into per-dimension indices
            size_t a_idx = 0, b_idx = 0, rem = idx;
            for (size_t d = 0; d < rank; ++d) {
                size_t coord = rem / o_strides[d];
                rem %= o_strides[d];
                a_idx += (sa[d] == 1 ? 0 : coord) * a_strides[d];
                b_idx += (sb[d] == 1 ? 0 : coord) * b_strides[d];
            }
            float va = a[a_idx], vb = b[b_idx];
            switch (mode) {
                case cldnn::eltwise_mode::sum:  output[idx] = va + vb; break;
                case cldnn::eltwise_mode::sub:  output[idx] = va - vb; break;
                case cldnn::eltwise_mode::prod: output[idx] = va * vb; break;
                case cldnn::eltwise_mode::div:  output[idx] = (vb != 0.0f) ? va / vb : 0.0f; break;
                case cldnn::eltwise_mode::max:  output[idx] = std::max(va, vb); break;
                case cldnn::eltwise_mode::min:  output[idx] = std::min(va, vb); break;
                case cldnn::eltwise_mode::pow:  output[idx] = std::pow(va, vb); break;
                case cldnn::eltwise_mode::squared_diff: output[idx] = (va - vb) * (va - vb); break;
                case cldnn::eltwise_mode::mod:  output[idx] = (vb != 0.0f) ? std::fmod(va, vb) : 0.0f; break;
                case cldnn::eltwise_mode::eq:   output[idx] = (va == vb) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::ne:   output[idx] = (va != vb) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::lt:   output[idx] = (va <  vb) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::le:   output[idx] = (va <= vb) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::gt:   output[idx] = (va >  vb) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::ge:   output[idx] = (va >= vb) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::logic_and: output[idx] = (va != 0.0f && vb != 0.0f) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::logic_or:  output[idx] = (va != 0.0f || vb != 0.0f) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::logic_xor: output[idx] = ((va != 0.0f) != (vb != 0.0f)) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::floor_mod: output[idx] = (vb != 0.0f) ? (va - std::floor(va / vb) * vb) : 0.0f; break;
                case cldnn::eltwise_mode::is_finite: output[idx] = std::isfinite(va) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::is_inf:    output[idx] = std::isinf(va) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::is_nan:    output[idx] = std::isnan(va) ? 1.0f : 0.0f; break;
                case cldnn::eltwise_mode::right_shift: output[idx] = static_cast<float>(static_cast<int32_t>(va) >> static_cast<int32_t>(vb)); break;
                case cldnn::eltwise_mode::left_shift:  output[idx] = static_cast<float>(static_cast<int32_t>(va) << static_cast<int32_t>(vb)); break;
                case cldnn::eltwise_mode::bitwise_and: output[idx] = static_cast<float>(static_cast<int32_t>(va) & static_cast<int32_t>(vb)); break;
                case cldnn::eltwise_mode::bitwise_or:  output[idx] = static_cast<float>(static_cast<int32_t>(va) | static_cast<int32_t>(vb)); break;
                case cldnn::eltwise_mode::bitwise_xor: output[idx] = static_cast<float>(static_cast<int32_t>(va) ^ static_cast<int32_t>(vb)); break;
                default: output[idx] = va + vb; break;
            }
        }
        return output;
    }

    // Fallback: same-shape element-wise (or simple flat broadcast)
    size_t n = std::max(a.size(), b.size());
    std::vector<float> output(n);
    for (size_t i = 0; i < n; ++i) {
        float va = a[i % a.size()];
        float vb = b[i % b.size()];
        switch (mode) {
            case cldnn::eltwise_mode::sum:  output[i] = va + vb; break;
            case cldnn::eltwise_mode::sub:  output[i] = va - vb; break;
            case cldnn::eltwise_mode::prod: output[i] = va * vb; break;
            case cldnn::eltwise_mode::div:  output[i] = (vb != 0.0f) ? va / vb : 0.0f; break;
            case cldnn::eltwise_mode::max:  output[i] = std::max(va, vb); break;
            case cldnn::eltwise_mode::min:  output[i] = std::min(va, vb); break;
            case cldnn::eltwise_mode::pow:  output[i] = std::pow(va, vb); break;
            case cldnn::eltwise_mode::squared_diff: output[i] = (va - vb) * (va - vb); break;
            case cldnn::eltwise_mode::mod:  output[i] = (vb != 0.0f) ? std::fmod(va, vb) : 0.0f; break;
            case cldnn::eltwise_mode::eq:   output[i] = (va == vb) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::ne:   output[i] = (va != vb) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::lt:   output[i] = (va <  vb) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::le:   output[i] = (va <= vb) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::gt:   output[i] = (va >  vb) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::ge:   output[i] = (va >= vb) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::logic_and: output[i] = (va != 0.0f && vb != 0.0f) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::logic_or:  output[i] = (va != 0.0f || vb != 0.0f) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::logic_xor: output[i] = ((va != 0.0f) != (vb != 0.0f)) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::floor_mod: output[i] = (vb != 0.0f) ? (va - std::floor(va / vb) * vb) : 0.0f; break;
            case cldnn::eltwise_mode::is_finite: output[i] = std::isfinite(va) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::is_inf:    output[i] = std::isinf(va) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::is_nan:    output[i] = std::isnan(va) ? 1.0f : 0.0f; break;
            case cldnn::eltwise_mode::right_shift: output[i] = static_cast<float>(static_cast<int32_t>(va) >> static_cast<int32_t>(vb)); break;
            case cldnn::eltwise_mode::left_shift:  output[i] = static_cast<float>(static_cast<int32_t>(va) << static_cast<int32_t>(vb)); break;
            case cldnn::eltwise_mode::bitwise_and: output[i] = static_cast<float>(static_cast<int32_t>(va) & static_cast<int32_t>(vb)); break;
            case cldnn::eltwise_mode::bitwise_or:  output[i] = static_cast<float>(static_cast<int32_t>(va) | static_cast<int32_t>(vb)); break;
            case cldnn::eltwise_mode::bitwise_xor: output[i] = static_cast<float>(static_cast<int32_t>(va) ^ static_cast<int32_t>(vb)); break;
            default: output[i] = va + vb; break;
        }
    }
    return output;
}

// --- Reduce ---
inline std::vector<float> reduce(const std::vector<float>& input,
                                  const std::vector<int64_t>& shape,
                                  cldnn::reduce_mode mode,
                                  const std::vector<int64_t>& axes,
                                  bool keep_dims) {
    size_t rank = shape.size();
    size_t total = total_elements(shape);

    // Build reduced-axis set
    std::vector<bool> is_reduced(rank, false);
    for (auto a : axes) {
        int64_t ax = a < 0 ? a + static_cast<int64_t>(rank) : a;
        is_reduced[static_cast<size_t>(ax)] = true;
    }

    // Compute output shape
    std::vector<int64_t> out_shape;
    for (size_t d = 0; d < rank; ++d) {
        if (is_reduced[d]) {
            if (keep_dims) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[d]);
        }
    }
    size_t out_total = total_elements(out_shape);

    // Init output
    std::vector<float> output(out_total);
    std::vector<size_t> count(out_total, 0);
    for (size_t i = 0; i < out_total; ++i) {
        switch (mode) {
            case cldnn::reduce_mode::sum:
            case cldnn::reduce_mode::mean:
            case cldnn::reduce_mode::l1:
            case cldnn::reduce_mode::sum_square:
            case cldnn::reduce_mode::l2:
                output[i] = 0.0f; break;
            case cldnn::reduce_mode::max:
                output[i] = -1e30f; break;
            case cldnn::reduce_mode::min:
                output[i] = 1e30f; break;
            case cldnn::reduce_mode::prod:
                output[i] = 1.0f; break;
            default:
                output[i] = 0.0f; break;
        }
    }

    // Compute strides for input shape
    std::vector<size_t> strides(rank);
    strides[rank - 1] = 1;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * static_cast<size_t>(shape[d + 1]);

    // Iterate all input elements
    std::vector<int64_t> coords(rank);
    std::vector<int64_t> out_coords_r;
    out_coords_r.reserve(rank);
    for (size_t idx = 0; idx < total; ++idx) {
        // Decompose flat index into multi-dim coords
        size_t remaining = idx;
        for (size_t d = 0; d < rank; ++d) {
            coords[d] = static_cast<int64_t>(remaining / strides[d]);
            remaining %= strides[d];
        }
        // Map input coords to output coords and compute flat output index
        size_t oi = 0;
        {
            out_coords_r.clear();
            for (size_t d = 0; d < rank; ++d) {
                if (is_reduced[d]) {
                    if (keep_dims) out_coords_r.push_back(0);
                } else {
                    out_coords_r.push_back(coords[d]);
                }
            }
            size_t omul = 1;
            for (int d = static_cast<int>(out_coords_r.size()) - 1; d >= 0; --d) {
                oi += static_cast<size_t>(out_coords_r[d]) * omul;
                omul *= static_cast<size_t>(out_shape[d]);
            }
        }

        float v = input[idx];
        switch (mode) {
            case cldnn::reduce_mode::sum:
            case cldnn::reduce_mode::mean:
                output[oi] += v; break;
            case cldnn::reduce_mode::max:
                output[oi] = std::max(output[oi], v); break;
            case cldnn::reduce_mode::min:
                output[oi] = std::min(output[oi], v); break;
            case cldnn::reduce_mode::prod:
                output[oi] *= v; break;
            case cldnn::reduce_mode::l1:
                output[oi] += std::fabs(v); break;
            case cldnn::reduce_mode::sum_square:
                output[oi] += v * v; break;
            case cldnn::reduce_mode::l2:
                output[oi] += v * v; break;
            default:
                output[oi] += v; break;
        }
        count[oi]++;
    }

    // For mean, divide by count
    if (mode == cldnn::reduce_mode::mean) {
        for (size_t i = 0; i < out_total; ++i) {
            if (count[i] > 0) output[i] /= static_cast<float>(count[i]);
        }
    }
    // For l2, take square root
    if (mode == cldnn::reduce_mode::l2) {
        for (size_t i = 0; i < out_total; ++i) {
            output[i] = std::sqrt(output[i]);
        }
    }
    return output;
}

// --- Pooling (2D, max/avg, configurable kernel/stride) ---
inline std::vector<float> pooling2d(const std::vector<float>& input,
                                     const std::vector<int64_t>& shape,
                                     bool is_max,
                                     int64_t kH = 2, int64_t kW = 2,
                                     int64_t sH = 2, int64_t sW = 2) {
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int64_t OH = (H - kH) / sH + 1;
    int64_t OW = (W - kW) / sW + 1;

    std::vector<float> output(static_cast<size_t>(N * C * OH * OW));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float val = is_max ? -1e30f : 0.0f;
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t ih = oh * sH + kh;
                            int64_t iw = ow * sW + kw;
                            float v = input[((n * C + c) * H + ih) * W + iw];
                            if (is_max) val = std::max(val, v);
                            else val += v;
                        }
                    }
                    if (!is_max) val /= static_cast<float>(kH * kW);
                    output[((n * C + c) * OH + oh) * OW + ow] = val;
                }
            }
        }
    }
    return output;
}

// --- Generic N-D Pooling (max/avg) with optional padding ---
// input: [N, C, spatial...], kernel/stride arrays match spatial dims
// pool_mode: 0=max, 1=average (count includes padding), 2=average_no_padding (count excludes padding)
// max_init: initial value for max pooling (use -65504 for f16, numeric_limits::lowest() for f32)
inline std::vector<float> poolingNd(const std::vector<float>& input,
                                     const std::vector<int64_t>& shape,
                                     int pool_mode,
                                     const std::vector<int64_t>& kernel,
                                     const std::vector<int64_t>& stride,
                                     const std::vector<int64_t>& pads_begin_in = {},
                                     const std::vector<int64_t>& pads_end_in = {},
                                     bool ceil_mode = false,
                                     float max_init = -65504.0f) {
    bool is_max = (pool_mode == 0);
    bool avg_include_pad = (pool_mode == 1);
    size_t rank = shape.size();
    size_t spatial_dims = rank - 2;
    int64_t N = shape[0], C = shape[1];

    std::vector<int64_t> pb(spatial_dims, 0);
    std::vector<int64_t> pe(spatial_dims, 0);
    for (size_t d = 0; d < spatial_dims; ++d) {
        if (d < pads_begin_in.size()) pb[d] = pads_begin_in[d];
        if (d < pads_end_in.size()) pe[d] = pads_end_in[d];
    }

    std::vector<int64_t> out_spatial(spatial_dims);
    int64_t out_spatial_total = 1;
    int64_t kernel_total = 1;
    for (size_t d = 0; d < spatial_dims; ++d) {
        int64_t numerator = shape[d + 2] + pb[d] + pe[d] - kernel[d];
        if (ceil_mode) {
            out_spatial[d] = (numerator + stride[d] - 1) / stride[d] + 1;
        } else {
            out_spatial[d] = numerator / stride[d] + 1;
        }
        out_spatial_total *= out_spatial[d];
        kernel_total *= kernel[d];
    }

    std::vector<float> output(static_cast<size_t>(N * C * out_spatial_total));

    // Pre-allocate coord buffers outside loops
    std::vector<int64_t> out_coords(spatial_dims);
    std::vector<int64_t> k_coords(spatial_dims);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t os = 0; os < out_spatial_total; ++os) {
                {
                    int64_t rem = os;
                    for (int sd = static_cast<int>(spatial_dims) - 1; sd >= 0; --sd) {
                        out_coords[sd] = rem % out_spatial[sd];
                        rem /= out_spatial[sd];
                    }
                }

                float val = is_max ? max_init : 0.0f;
                int64_t count = 0;
                for (int64_t ki = 0; ki < kernel_total; ++ki) {
                    {
                        int64_t rem = ki;
                        for (int sd = static_cast<int>(spatial_dims) - 1; sd >= 0; --sd) {
                            k_coords[sd] = rem % kernel[sd];
                            rem /= kernel[sd];
                        }
                    }

                    // Compute input position with padding
                    bool in_bounds = true;
                    size_t in_idx = static_cast<size_t>(n);
                    in_idx = in_idx * static_cast<size_t>(C) + static_cast<size_t>(c);
                    for (size_t d = 0; d < spatial_dims; ++d) {
                        int64_t pos = out_coords[d] * stride[d] - pb[d] + k_coords[d];
                        if (pos < 0 || pos >= shape[d + 2]) { in_bounds = false; break; }
                        in_idx = in_idx * static_cast<size_t>(shape[d + 2])
                               + static_cast<size_t>(pos);
                    }
                    if (!in_bounds) continue;

                    float v = input[in_idx];
                    if (is_max) val = std::max(val, v);
                    else val += v;
                    count++;
                }
                // For avg_include_pad (mode=1): divide by kernel_total
                // For avg_no_padding (mode=2): divide by actual valid count
                if (!is_max && count > 0) {
                    val /= static_cast<float>(avg_include_pad ? kernel_total : count);
                }

                size_t out_flat = static_cast<size_t>(n);
                out_flat = out_flat * static_cast<size_t>(C) + static_cast<size_t>(c);
                for (size_t d = 0; d < spatial_dims; ++d) {
                    out_flat = out_flat * static_cast<size_t>(out_spatial[d])
                             + static_cast<size_t>(out_coords[d]);
                }
                output[out_flat] = val;
            }
        }
    }
    return output;
}

// --- MVN (mean-variance normalization) ---
// Normalize across all dims except batch
// eps_inside_sqrt: if true, inv_std = 1/sqrt(var + eps); if false, inv_std = 1/(sqrt(var) + eps)
inline std::vector<float> mvn(const std::vector<float>& input,
                               const std::vector<int64_t>& shape,
                               bool normalize_variance,
                               float epsilon,
                               bool eps_inside_sqrt = true) {
    int64_t batch = shape[0];
    size_t inner = total_elements(shape) / static_cast<size_t>(batch);

    std::vector<float> output(input.size());
    for (int64_t b = 0; b < batch; ++b) {
        size_t offset = b * inner;
        // Mean
        float mean = 0.0f;
        for (size_t i = 0; i < inner; ++i) mean += input[offset + i];
        mean /= static_cast<float>(inner);

        if (!normalize_variance) {
            for (size_t i = 0; i < inner; ++i) output[offset + i] = input[offset + i] - mean;
        } else {
            // Variance
            float var = 0.0f;
            for (size_t i = 0; i < inner; ++i) {
                float diff = input[offset + i] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(inner);
            float inv_std;
            if (eps_inside_sqrt) {
                inv_std = 1.0f / std::sqrt(var + epsilon);
            } else {
                inv_std = 1.0f / (std::sqrt(var) + epsilon);
            }
            for (size_t i = 0; i < inner; ++i) {
                output[offset + i] = (input[offset + i] - mean) * inv_std;
            }
        }
    }
    return output;
}

// --- Permute ---
inline std::vector<float> permute(const std::vector<float>& input,
                                   const std::vector<int64_t>& shape,
                                   const std::vector<uint16_t>& order) {
    size_t rank = shape.size();
    size_t total = total_elements(shape);

    // Compute output shape
    std::vector<int64_t> out_shape(rank);
    for (size_t d = 0; d < rank; ++d) out_shape[d] = shape[order[d]];

    // Compute input strides
    std::vector<size_t> in_strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * static_cast<size_t>(shape[d + 1]);
    }

    // Compute output strides
    std::vector<size_t> out_strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
        out_strides[d] = out_strides[d + 1] * static_cast<size_t>(out_shape[d + 1]);
    }

    std::vector<float> output(total);
    std::vector<size_t> coords(rank, 0);

    for (size_t i = 0; i < total; ++i) {
        // Compute input flat index from output coords
        size_t in_idx = 0;
        for (size_t d = 0; d < rank; ++d) {
            in_idx += coords[d] * in_strides[order[d]];
        }
        output[i] = input[in_idx];

        // Increment output coords
        for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < static_cast<size_t>(out_shape[d])) break;
            coords[d] = 0;
        }
    }
    return output;
}

// --- Reorder (type conversion) ---
// GPU reorder_data.cl behavior:
//   The truncate/saturation flag ONLY affects float→integer conversions:
//     #if INPUT0_IS_FP && !OUTPUT_IS_FP
//       #if CONVERT_TRUNCATE → TO_OUTPUT_REORDER_TYPE(convert_long(res))   // wrapping
//       #else                → TO_OUTPUT_REORDER_TYPE_SAT(res)             // saturation
//     #else → TO_OUTPUT_REORDER_TYPE(res)                                  // plain conversion
//
//   truncate=true  (Convert op): float → convert_long → narrow (wrapping/modulo 2^N)
//   truncate=false (plain Reorder): float → convert_T_sat (saturation/clamp)
//   integer→integer narrowing: always plain conversion (truncation/wrapping)
//
// Helper to check if a data type is floating-point
inline bool is_fp_type(cldnn::data_types dt) {
    return dt == cldnn::data_types::f32 || dt == cldnn::data_types::f16;
}

inline std::vector<float> reorder(const std::vector<float>& input,
                                   cldnn::data_types src_dt = cldnn::data_types::f32,
                                   cldnn::data_types dst_dt = cldnn::data_types::f32,
                                   bool truncate = false) {
    // If no type narrowing, just pass through
    if (src_dt == dst_dt || dst_dt == cldnn::data_types::f32) {
        return input;
    }

    // Determine effective truncation mode based on GPU kernel logic:
    // - Float→integer: controlled by truncate flag
    // - Integer→integer: always plain conversion (truncation/wrapping)
    bool src_is_fp = is_fp_type(src_dt);
    bool dst_is_int = !is_fp_type(dst_dt);
    bool use_saturation = src_is_fp && dst_is_int && !truncate;

    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float v = input[i];
        switch (dst_dt) {
            case cldnn::data_types::u8: {
                if (use_saturation) {
                    // Saturation: clamp to [0, 255]
                    int32_t iv = static_cast<int32_t>(v);
                    iv = std::max(0, std::min(255, iv));
                    output[i] = static_cast<float>(static_cast<uint8_t>(iv));
                } else {
                    // Truncation: value → int32 (truncate toward zero) → uint8 (modulo 256)
                    int32_t iv = static_cast<int32_t>(v);
                    output[i] = static_cast<float>(static_cast<uint8_t>(iv));
                }
                break;
            }
            case cldnn::data_types::i8: {
                if (use_saturation) {
                    // Saturation: clamp to [-128, 127]
                    int32_t iv = static_cast<int32_t>(v);
                    iv = std::max(-128, std::min(127, iv));
                    output[i] = static_cast<float>(static_cast<int8_t>(iv));
                } else {
                    // Truncation: value → int32 → int8 (modulo 256)
                    int32_t iv = static_cast<int32_t>(v);
                    output[i] = static_cast<float>(static_cast<int8_t>(iv));
                }
                break;
            }
            case cldnn::data_types::i32: {
                double dv = static_cast<double>(v);
                dv = std::max(static_cast<double>(std::numeric_limits<int32_t>::min()),
                              std::min(static_cast<double>(std::numeric_limits<int32_t>::max()), dv));
                output[i] = static_cast<float>(static_cast<int32_t>(dv));
                break;
            }
            case cldnn::data_types::f16: {
                // Simulate f16 precision loss via round-trip
                ov::float16 h = ov::float16(v);
                output[i] = static_cast<float>(h);
                break;
            }
            default:
                output[i] = v;
                break;
        }
    }
    return output;
}

// --- Concatenation along axis ---
inline std::vector<float> concat(const std::vector<std::vector<float>>& inputs,
                                  const std::vector<std::vector<int64_t>>& shapes,
                                  int64_t axis) {
    // Simplification: all inputs have same shape except along concat axis
    // Compute output size
    size_t total_out = 0;
    for (auto& inp : inputs) total_out += inp.size();

    // For simple concat along an axis, we need to interleave correctly
    // Compute the outer/inner strides
    auto& first_shape = shapes[0];
    size_t rank = first_shape.size();
    if (axis < 0) axis += static_cast<int64_t>(rank);

    size_t outer = 1;
    for (int64_t d = 0; d < axis; ++d) outer *= static_cast<size_t>(first_shape[d]);

    size_t inner = 1;
    for (size_t d = static_cast<size_t>(axis) + 1; d < rank; ++d) inner *= static_cast<size_t>(first_shape[d]);

    std::vector<float> output;
    output.reserve(total_out);

    for (size_t o = 0; o < outer; ++o) {
        for (size_t inp_idx = 0; inp_idx < inputs.size(); ++inp_idx) {
            size_t axis_size = static_cast<size_t>(shapes[inp_idx][axis]);
            size_t chunk = axis_size * inner;
            size_t src_offset = o * chunk;
            for (size_t i = 0; i < chunk; ++i) {
                output.push_back(inputs[inp_idx][src_offset + i]);
            }
        }
    }
    return output;
}

// --- SDPA: Scaled Dot-Product Attention with transpose orders ---
// Input tensors come in physical layout. Transpose orders specify how to
// interpret them as logical [B, H, S, D].
// order_q/k/v: maps physical dims to logical [B,H,S,D].
//   Physical[order[0]] = B, Physical[order[1]] = H, Physical[order[2]] = S, Physical[order[3]] = D
//   i.e. order specifies which physical dim corresponds to each logical dim
// order_out: maps logical [B,H,S,D] to output physical layout.
//
// Most common: identity {0,1,2,3} means physical=logical [B,H,S,D]
//              {0,2,1,3} means physical [B,S,H,D] → logical after transpose [B,H,S,D]
inline std::vector<float> sdpa(const std::vector<float>& Q,
                                const std::vector<int64_t>& Q_shape,
                                const std::vector<float>& K,
                                const std::vector<int64_t>& K_shape,
                                const std::vector<float>& V,
                                const std::vector<int64_t>& V_shape,
                                bool is_causal,
                                const std::vector<int64_t>& order_q = {0,1,2,3},
                                const std::vector<int64_t>& order_k = {0,1,2,3},
                                const std::vector<int64_t>& order_v = {0,1,2,3},
                                const std::vector<int64_t>& order_out = {0,1,2,3},
                                float scale_override = 0.0f,
                                const std::vector<float>* mask = nullptr,
                                const std::vector<int64_t>* mask_shape = nullptr) {
    // Helper: compute physical strides for a shape
    auto get_strides = [](const std::vector<int64_t>& sh) {
        std::vector<int64_t> s(sh.size(), 1);
        for (int i = static_cast<int>(sh.size()) - 2; i >= 0; --i)
            s[i] = s[i + 1] * sh[i + 1];
        return s;
    };

    // Compute logical dimensions via transpose orders
    // order[logical_dim] = physical_dim
    // so logical_shape[logical_dim] = physical_shape[order[logical_dim]]
    auto logical_dim = [](const std::vector<int64_t>& phys_shape, const std::vector<int64_t>& order, int dim) -> int64_t {
        return phys_shape[order[dim]];
    };

    // Logical Q: [B, Hq, Sq, D]
    int64_t B  = logical_dim(Q_shape, order_q, 0);
    int64_t Hq = logical_dim(Q_shape, order_q, 1);
    int64_t Sq = logical_dim(Q_shape, order_q, 2);
    int64_t D  = logical_dim(Q_shape, order_q, 3);
    // Logical K: [B, Hkv, Sk, D]
    int64_t Hkv = logical_dim(K_shape, order_k, 1);
    int64_t Sk  = logical_dim(K_shape, order_k, 2);

    float scale = (scale_override != 0.0f) ? scale_override : (1.0f / std::sqrt(static_cast<float>(D)));

    // Physical strides
    auto q_strides = get_strides(Q_shape);
    auto k_strides = get_strides(K_shape);
    auto v_strides = get_strides(V_shape);

    // Helper: access element by logical [b,h,s,d] using physical layout + order
    auto q_idx = [&](int64_t b, int64_t h, int64_t s, int64_t d) -> size_t {
        int64_t coords[4] = {b, h, s, d};
        int64_t phys = 0;
        for (int i = 0; i < 4; ++i)
            phys += coords[i] * q_strides[order_q[i]];
        return static_cast<size_t>(phys);
    };
    auto k_idx = [&](int64_t b, int64_t h, int64_t s, int64_t d) -> size_t {
        int64_t coords[4] = {b, h, s, d};
        int64_t phys = 0;
        for (int i = 0; i < 4; ++i)
            phys += coords[i] * k_strides[order_k[i]];
        return static_cast<size_t>(phys);
    };
    auto v_idx = [&](int64_t b, int64_t h, int64_t s, int64_t d) -> size_t {
        int64_t coords[4] = {b, h, s, d};
        int64_t phys = 0;
        for (int i = 0; i < 4; ++i)
            phys += coords[i] * v_strides[order_v[i]];
        return static_cast<size_t>(phys);
    };

    // Output: logical [B, Hq, Sq, D] always in [B,H,S,D] physical order
    // The GPU SDPA kernel always writes in [B,H,S,D] memory order; order_out
    // only changes shape metadata for downstream consumers, not physical layout.
    size_t out_total = static_cast<size_t>(B * Hq * Sq * D);
    // Mask support: compute mask strides for numpy broadcast
    // mask shape is typically [N, ..., L, S] and is broadcast to [B, Hq, Sq, Sk]
    std::vector<int64_t> mask_strides_bc;  // broadcast-aware strides
    if (mask && mask_shape && !mask->empty()) {
        auto& ms = *mask_shape;
        size_t mrank = ms.size();
        // Compute physical strides
        std::vector<int64_t> ms_strides(mrank, 1);
        for (int i = static_cast<int>(mrank) - 2; i >= 0; --i)
            ms_strides[i] = ms_strides[i + 1] * ms[i + 1];
        // Broadcast to logical [B, Hq, Sq, Sk] - pad mask shape to 4D from left
        mask_strides_bc.resize(4, 0);
        for (int i = 0; i < 4; ++i) {
            int mi = static_cast<int>(mrank) - 4 + i;  // index into mask shape
            if (mi >= 0 && ms[mi] > 1) {
                mask_strides_bc[i] = ms_strides[mi];
            } else {
                mask_strides_bc[i] = 0;  // broadcast dimension
            }
        }
    }

    auto get_mask_val = [&](int64_t b, int64_t h, int64_t sq_pos, int64_t sk_pos) -> float {
        if (mask_strides_bc.empty()) return 0.0f;
        size_t idx = static_cast<size_t>(b * mask_strides_bc[0] + h * mask_strides_bc[1] +
                                         sq_pos * mask_strides_bc[2] + sk_pos * mask_strides_bc[3]);
        return (*mask)[idx];
    };

    std::vector<float> output(out_total, 0.0f);

    auto out_idx = [&](int64_t b, int64_t h, int64_t s, int64_t d) -> size_t {
        return static_cast<size_t>(((b * Hq + h) * Sq + s) * D + d);
    };

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < Hq; ++h) {
            int64_t kv_h = (Hkv == Hq) ? h : (h / (Hq / Hkv));

            for (int64_t sq = 0; sq < Sq; ++sq) {
                std::vector<float> scores(static_cast<size_t>(Sk));
                for (int64_t sk = 0; sk < Sk; ++sk) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        dot += Q[q_idx(b, h, sq, d)] * K[k_idx(b, kv_h, sk, d)];
                    }
                    scores[sk] = dot * scale;
                    if (is_causal && sk > sq) {
                        scores[sk] = -1e9f;
                    } else if (!is_causal && mask && !mask_strides_bc.empty()) {
                        // Per OpenVINO SDPA spec: attn_weight += attn_mask
                        scores[sk] += get_mask_val(b, h, sq, sk);
                    }
                }

                float max_s = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int64_t sk = 0; sk < Sk; ++sk) {
                    scores[sk] = std::exp(scores[sk] - max_s);
                    sum_exp += scores[sk];
                }
                for (int64_t sk = 0; sk < Sk; ++sk) {
                    scores[sk] /= sum_exp;
                }

                for (int64_t d = 0; d < D; ++d) {
                    float val = 0.0f;
                    for (int64_t sk = 0; sk < Sk; ++sk) {
                        val += scores[sk] * V[v_idx(b, kv_h, sk, d)];
                    }
                    output[out_idx(b, h, sq, d)] = val;
                }
            }
        }
    }
    return output;
}

// ============================================================================
// RMS Normalization reference
// output[i] = input[i] / sqrt(mean(input[...,:]^2) + eps) * gamma[i]
// Normalization is across the last dimension
// ============================================================================
inline std::vector<float> rms_norm(const std::vector<float>& input,
                                   const std::vector<float>& gamma,
                                   const std::vector<int64_t>& shape,
                                   float epsilon) {
    size_t total = input.size();
    int64_t hidden_dim = shape.back();
    size_t num_tokens = total / hidden_dim;
    std::vector<float> output(total);

    for (size_t t = 0; t < num_tokens; ++t) {
        const float* inp = input.data() + t * hidden_dim;
        float* out = output.data() + t * hidden_dim;

        // Compute mean of squares
        double sum_sq = 0.0;
        for (int64_t i = 0; i < hidden_dim; ++i) {
            sum_sq += static_cast<double>(inp[i]) * static_cast<double>(inp[i]);
        }
        float rms = std::sqrt(static_cast<float>(sum_sq / hidden_dim) + epsilon);

        for (int64_t i = 0; i < hidden_dim; ++i) {
            out[i] = (inp[i] / rms) * gamma[i];
        }
    }
    return output;
}

// ============================================================================
// SwiGLU reference (configurable)
// Splits input along split_axis into two halves (or by split_length):
//   gate = first part, up = second part (or swapped if gate_idx=1)
// Applies gating function: output = gate_fn(gate) * up
// glu_type: 0=Swish, 1=Gelu, 2=GeluTanh
// ============================================================================
inline std::vector<float> swiglu(const std::vector<float>& input,
                                 const std::vector<int64_t>& shape,
                                 int64_t split_axis = -1,
                                 int64_t split_length = -1,
                                 int glu_type = 0,
                                 int gate_idx = 0) {
    size_t rank = shape.size();
    if (split_axis < 0) split_axis += static_cast<int64_t>(rank);
    int64_t axis_dim = shape[split_axis];
    if (split_length < 0) split_length = axis_dim / 2;

    // Compute outer/inner sizes around split_axis
    size_t outer = 1;
    for (int64_t d = 0; d < split_axis; ++d) outer *= static_cast<size_t>(shape[d]);
    size_t inner = 1;
    for (size_t d = static_cast<size_t>(split_axis) + 1; d < rank; ++d) inner *= static_cast<size_t>(shape[d]);

    size_t out_total = outer * static_cast<size_t>(split_length) * inner;
    std::vector<float> output(out_total);

    for (size_t o = 0; o < outer; ++o) {
        for (int64_t s = 0; s < split_length; ++s) {
            for (size_t i = 0; i < inner; ++i) {
                // gate part: [0..split_length), up part: [split_length..axis_dim)
                size_t gate_offset, up_offset;
                if (gate_idx == 0) {
                    gate_offset = (o * static_cast<size_t>(axis_dim) + static_cast<size_t>(s)) * inner + i;
                    up_offset = (o * static_cast<size_t>(axis_dim) + static_cast<size_t>(split_length + s)) * inner + i;
                } else {
                    // gate is second half
                    up_offset = (o * static_cast<size_t>(axis_dim) + static_cast<size_t>(s)) * inner + i;
                    gate_offset = (o * static_cast<size_t>(axis_dim) + static_cast<size_t>(split_length + s)) * inner + i;
                }

                float gate = input[gate_offset];
                float up = input[up_offset];

                // Apply gating function
                float gated;
                switch (glu_type) {
                    case 1:  // Gelu
                        gated = gate * 0.5f * (1.0f + std::erf(gate / std::sqrt(2.0f)));
                        break;
                    case 2: {  // GeluTanh
                        float k = std::sqrt(2.0f / 3.14159265358979f);
                        gated = 0.5f * gate * (1.0f + std::tanh(k * (gate + 0.044715f * gate * gate * gate)));
                        break;
                    }
                    default:  // 0 = Swish
                        gated = gate / (1.0f + std::exp(-gate));
                        break;
                }
                size_t out_offset = (o * static_cast<size_t>(split_length) + static_cast<size_t>(s)) * inner + i;
                output[out_offset] = gated * up;
            }
        }
    }
    return output;
}

// ============================================================================
// Gather reference
// Gathers elements from dict along axis using indices
// ============================================================================
inline std::vector<float> gather(const std::vector<float>& dict,
                                 const std::vector<int32_t>& indices,
                                 const std::vector<int64_t>& dict_shape,
                                 const std::vector<int64_t>& idx_shape,
                                 int64_t axis) {
    int64_t rank = static_cast<int64_t>(dict_shape.size());
    if (axis < 0) axis += rank;

    // Compute strides for dict
    std::vector<int64_t> dict_strides(rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i)
        dict_strides[i] = dict_strides[i + 1] * dict_shape[i + 1];

    // Compute output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < axis; ++i) out_shape.push_back(dict_shape[i]);
    for (auto d : idx_shape) out_shape.push_back(d);
    for (int64_t i = axis + 1; i < rank; ++i) out_shape.push_back(dict_shape[i]);

    size_t out_total = 1;
    for (auto d : out_shape) out_total *= d;

    // Compute sizes for the three parts: before_axis, idx, after_axis
    int64_t before = 1;
    for (int64_t i = 0; i < axis; ++i) before *= dict_shape[i];
    int64_t after = 1;
    for (int64_t i = axis + 1; i < rank; ++i) after *= dict_shape[i];
    int64_t num_idx = static_cast<int64_t>(indices.size());

    std::vector<float> output(out_total);

    if (bench_ref_parallel_enabled() && (before * num_idx * after) >= 1024) {
        tbb::parallel_for(tbb::blocked_range<int64_t>(0, before * num_idx * after),
            [&](const tbb::blocked_range<int64_t>& r) {
            for (int64_t flat_idx = r.begin(); flat_idx != r.end(); ++flat_idx) {
                int64_t b = flat_idx / (num_idx * after);
                int64_t remainder = flat_idx % (num_idx * after);
                int64_t idx_i = remainder / after;
                int64_t a = remainder % after;

                int32_t idx_val = indices[idx_i];
                // Normalize negative indices per OpenVINO Gather-8 spec
                if (idx_val < 0) idx_val += static_cast<int32_t>(dict_shape[axis]);
                // Out-of-range indices → output filled with 0 (per spec)
                if (idx_val < 0 || idx_val >= static_cast<int32_t>(dict_shape[axis])) {
                    int64_t out_offset = b * num_idx * after + idx_i * after + a;
                    output[out_offset] = 0.0f;
                } else {
                    int64_t dict_offset = b * dict_shape[axis] * after + idx_val * after + a;
                    int64_t out_offset = b * num_idx * after + idx_i * after + a;
                    output[out_offset] = dict[dict_offset];
                }
            }
        });
    } else {
        for (int64_t b = 0; b < before; ++b) {
            for (int64_t idx_i = 0; idx_i < num_idx; ++idx_i) {
                int32_t idx_val = indices[idx_i];
                // Normalize negative indices per OpenVINO Gather-8 spec
                if (idx_val < 0) idx_val += static_cast<int32_t>(dict_shape[axis]);
                // Out-of-range indices → output filled with 0 (per spec)
                if (idx_val < 0 || idx_val >= static_cast<int32_t>(dict_shape[axis])) {
                    for (int64_t a = 0; a < after; ++a) {
                        int64_t out_offset = b * num_idx * after + idx_i * after + a;
                        output[out_offset] = 0.0f;
                    }
                    continue;
                }
                for (int64_t a = 0; a < after; ++a) {
                    int64_t dict_offset = b * dict_shape[axis] * after + idx_val * after + a;
                    int64_t out_offset = b * num_idx * after + idx_i * after + a;
                    output[out_offset] = dict[dict_offset];
                }
            }
        }
    }
    return output;
}

// ============================================================================
// Gather reference directly from GPU memory (axis=0, before=1 fast path)
// Avoids materializing the full dict as float — critical for large embedding
// tables (e.g., 128256x4096 u8 = 524 MB → 2 GB f32).
// Falls back to the full-dict gather() for non-trivial axis or before > 1.
// ============================================================================
inline std::vector<float> gather_ref_from_mem(
    cldnn::memory::ptr dict_mem,
    cldnn::stream& stream,
    const std::vector<int64_t>& dict_shape,
    cldnn::data_types dict_dt,
    const std::vector<int32_t>& indices,
    const std::vector<int64_t>& idx_shape,
    int64_t axis) {
    int64_t rank = static_cast<int64_t>(dict_shape.size());
    if (axis < 0) axis += rank;

    int64_t before = 1;
    for (int64_t i = 0; i < axis; ++i) before *= dict_shape[i];
    int64_t after = 1;
    for (int64_t i = axis + 1; i < rank; ++i) after *= dict_shape[i];
    int64_t axis_dim = dict_shape[axis];
    int64_t num_idx = static_cast<int64_t>(indices.size());

    // Only use targeted (sparse) path for axis=0 with before=1 (typical embedding lookup)
    if (before != 1) {
        auto dict_f32 = read_memory_to_f32(dict_mem, stream);
        return gather(dict_f32, indices, dict_shape, idx_shape, axis);
    }

    // Copy device memory to host if necessary
    cldnn::memory::ptr host_mem = dict_mem;
    if (dict_mem->get_allocation_type() == cldnn::allocation_type::usm_device) {
        auto* eng = dict_mem->get_engine();
        host_mem = eng->allocate_memory(dict_mem->get_layout(), cldnn::allocation_type::usm_host);
        host_mem->copy_from(stream, *dict_mem, true);
    }

    std::vector<float> ref_out(static_cast<size_t>(num_idx * after), 0.0f);

    // Read only the gathered rows from the dict using a type-specific lock
    // Each row has `after` elements; row r starts at flat index r * after
    auto do_gather = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto lock = cldnn::mem_lock<T>(host_mem, stream);
        if (bench_ref_parallel_enabled() && num_idx * after >= 1024) {
            tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_idx),
                [&](const tbb::blocked_range<int64_t>& r) {
                for (int64_t ii = r.begin(); ii != r.end(); ++ii) {
                    int32_t idx = indices[ii];
                    if (idx < 0) idx += static_cast<int32_t>(axis_dim);
                    if (idx < 0 || idx >= static_cast<int32_t>(axis_dim)) continue;
                    int64_t off = idx * after;
                    for (int64_t j = 0; j < after; ++j)
                        ref_out[ii * after + j] = static_cast<float>(lock[off + j]);
                }
            });
        } else {
            for (int64_t ii = 0; ii < num_idx; ++ii) {
                int32_t idx = indices[ii];
                if (idx < 0) idx += static_cast<int32_t>(axis_dim);
                if (idx < 0 || idx >= static_cast<int32_t>(axis_dim)) continue;
                int64_t off = idx * after;
                for (int64_t j = 0; j < after; ++j)
                    ref_out[ii * after + j] = static_cast<float>(lock[off + j]);
            }
        }
    };

    switch (dict_dt) {
        case cldnn::data_types::f32:  do_gather(float{});           break;
        case cldnn::data_types::f16:  do_gather(ov::float16{});     break;
        case cldnn::data_types::i8:   do_gather(int8_t{});          break;
        case cldnn::data_types::u8:   do_gather(uint8_t{});         break;
        case cldnn::data_types::i32:  do_gather(int32_t{});         break;
        case cldnn::data_types::i64:  do_gather(int64_t{});         break;
        default: {
            // Fallback for unusual types: full materialization
            auto dict_f32 = read_memory_to_f32(host_mem, stream);
            return gather(dict_f32, indices, dict_shape, idx_shape, axis);
        }
    }
    return ref_out;
}

// ============================================================================
// Crop (Slice) reference
// Crops from given offset with given crop_shape
// ============================================================================
inline std::vector<float> crop(const std::vector<float>& input,
                               const std::vector<int64_t>& input_shape,
                               const std::vector<int64_t>& crop_shape,
                               const std::vector<int64_t>& offsets = {}) {
    size_t rank = input_shape.size();

    // Compute strides
    std::vector<int64_t> in_strides(rank, 1);
    std::vector<int64_t> out_strides(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * crop_shape[i + 1];
    }

    // Apply offsets
    std::vector<int64_t> off(rank, 0);
    for (size_t i = 0; i < std::min(offsets.size(), rank); ++i) {
        off[i] = offsets[i];
    }

    size_t out_total = 1;
    for (auto d : crop_shape) out_total *= d;

    std::vector<float> output(out_total);

    for (size_t out_i = 0; out_i < out_total; ++out_i) {
        // Convert flat index to multi-dim coords in crop space + offset
        size_t in_idx = 0;
        size_t rem = out_i;
        for (size_t d = 0; d < rank; ++d) {
            int64_t coord = rem / out_strides[d];
            rem %= out_strides[d];
            in_idx += (coord + off[d]) * in_strides[d];
        }
        output[out_i] = input[in_idx];
    }
    return output;
}

// ============================================================================
// StridedSlice reference
// Implements strided slice with begin, strides, and masks
// begin_mask[i]==1 means ignore begin[i] (use 0 for positive stride)
// end_mask[i]==1 means ignore end (use dim size for positive stride)
// ============================================================================
inline std::vector<float> strided_slice(const std::vector<float>& input,
                                        const std::vector<int64_t>& input_shape,
                                        const std::vector<int64_t>& output_shape,
                                        const std::vector<int64_t>& begin,
                                        const std::vector<int64_t>& strides,
                                        const std::vector<int64_t>& begin_mask,
                                        const std::vector<int64_t>& end_mask) {
    size_t rank = input_shape.size();

    // Compute effective begin per dimension (respecting begin_mask)
    std::vector<int64_t> eff_begin(rank, 0);
    for (size_t d = 0; d < rank; ++d) {
        if (d < begin_mask.size() && begin_mask[d] != 0) {
            eff_begin[d] = 0;  // mask=1 → ignore begin, use 0
        } else if (d < begin.size()) {
            eff_begin[d] = begin[d];
            // Handle negative indices
            if (eff_begin[d] < 0) eff_begin[d] += input_shape[d];
            eff_begin[d] = std::max<int64_t>(0, std::min(eff_begin[d], input_shape[d]));
        }
    }

    // Compute strides
    std::vector<int64_t> eff_strides(rank, 1);
    for (size_t d = 0; d < std::min(strides.size(), rank); ++d) {
        eff_strides[d] = strides[d] != 0 ? strides[d] : 1;
    }

    // Compute input strides
    std::vector<int64_t> in_strides(rank, 1);
    std::vector<int64_t> out_strides(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * output_shape[i + 1];
    }

    size_t out_total = 1;
    for (auto d : output_shape) out_total *= d;

    std::vector<float> output(out_total);
    size_t in_total = 1;
    for (auto d : input_shape) in_total *= d;

    for (size_t out_i = 0; out_i < out_total; ++out_i) {
        size_t in_idx = 0;
        size_t rem = out_i;
        bool valid = true;
        for (size_t d = 0; d < rank; ++d) {
            int64_t out_coord = rem / out_strides[d];
            rem %= out_strides[d];
            int64_t in_coord = eff_begin[d] + out_coord * eff_strides[d];
            if (in_coord < 0 || in_coord >= input_shape[d]) { valid = false; break; }
            in_idx += in_coord * in_strides[d];
        }
        output[out_i] = (valid && in_idx < in_total) ? input[in_idx] : 0.0f;
    }
    return output;
}

// ============================================================================
// RoPE RotateHalf reference
// Input: [B, S, H, D], Cos/Sin: [1, 1, S, D] (or compatible broadcast shape)
// Output: [B, H, S, D] if output_trans0213, else [B, S, H, D]
//
// For each position & head, for r in [0, half_rotary_ndims):
//   out[r]      = cos[s,r]      * in[r]      - sin[s,r]      * in[r+half]
//   out[r+half] = cos[s,r+half] * in[r+half] + sin[s,r+half] * in[r]
// For d >= rotary_ndims: passthrough
// ============================================================================
inline std::vector<float> rope_rotate_half(const std::vector<float>& input,
                                           const std::vector<float>& cos_table,
                                           const std::vector<float>& sin_table,
                                           const std::vector<int64_t>& input_shape,
                                           const std::vector<int64_t>& cos_shape,
                                           int64_t rotary_ndims,
                                           bool output_trans0213) {
    // input: [B, S, H, D]
    int64_t B = input_shape[0];
    int64_t S = input_shape[1];
    int64_t H = input_shape[2];
    int64_t D = input_shape[3];
    int64_t half_rot = rotary_ndims / 2;

    // cos/sin shape: typically [1, 1, S, D] or [1, S, 1, D] etc.
    // We index as: cos[0, 0, s, d] -> flat index = s * cos_last_dim + d
    int64_t cos_last_dim = cos_shape.back();

    // Compute cos/sin strides for safe broadcast indexing
    std::vector<int64_t> cs_strides(cos_shape.size(), 1);
    for (int i = static_cast<int>(cos_shape.size()) - 2; i >= 0; --i)
        cs_strides[i] = cs_strides[i + 1] * cos_shape[i + 1];

    auto get_cs_idx = [&](int64_t b, int64_t s_pos, int64_t d_pos) -> int64_t {
        // Map to cos/sin dims: [b_dim, s_dim_or_1, ..., d_pos]
        // Most common: 4D [1, 1, S, D] -> idx = s_pos * D + d_pos
        if (cos_shape.size() == 4) {
            int64_t cb = (b < cos_shape[0]) ? b : 0;
            int64_t c1 = 0;  // cos_shape[1] is typically 1
            int64_t cs = (s_pos < cos_shape[2]) ? s_pos : 0;
            return cb * cs_strides[0] + c1 * cs_strides[1] + cs * cs_strides[2] + d_pos;
        } else if (cos_shape.size() == 3) {
            int64_t cb = (b < cos_shape[0]) ? b : 0;
            int64_t cs = (s_pos < cos_shape[1]) ? s_pos : 0;
            return cb * cs_strides[0] + cs * cs_strides[1] + d_pos;
        } else {
            // 2D: [S, D]
            int64_t cs = (s_pos < cos_shape[0]) ? s_pos : 0;
            return cs * cos_last_dim + d_pos;
        }
    };

    // Output shape
    size_t out_total = B * H * S * D;
    std::vector<float> output(out_total);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            for (int64_t h = 0; h < H; ++h) {
                int64_t in_base = ((b * S + s) * H + h) * D;

                int64_t out_base;
                if (output_trans0213) {
                    // output[B, H, S, D]
                    out_base = ((b * H + h) * S + s) * D;
                } else {
                    // output[B, S, H, D]
                    out_base = ((b * S + s) * H + h) * D;
                }

                // Apply rotation to first rotary_ndims dimensions
                for (int64_t r = 0; r < half_rot; ++r) {
                    float in1 = input[in_base + r];
                    float in2 = input[in_base + half_rot + r];

                    float c1 = cos_table[get_cs_idx(b, s, r)];
                    float s1 = sin_table[get_cs_idx(b, s, r)];
                    float c2 = cos_table[get_cs_idx(b, s, half_rot + r)];
                    float s2 = sin_table[get_cs_idx(b, s, half_rot + r)];

                    output[out_base + r]          = c1 * in1 - s1 * in2;
                    output[out_base + half_rot + r] = c2 * in2 + s2 * in1;
                }

                // Passthrough for dimensions beyond rotary_ndims
                for (int64_t d = rotary_ndims; d < D; ++d) {
                    output[out_base + d] = input[in_base + d];
                }
            }
        }
    }
    return output;
}

// ============================================================================
// Broadcast reference (numpy-style)
// Broadcasts input_shape to target_shape
// ============================================================================
inline std::vector<float> broadcast(const std::vector<float>& input,
                                    const std::vector<int64_t>& input_shape,
                                    const std::vector<int64_t>& target_shape) {
    size_t out_rank = target_shape.size();
    size_t in_rank = input_shape.size();

    // Pad input shape with leading 1s to match output rank
    std::vector<int64_t> padded_in(out_rank, 1);
    for (size_t i = 0; i < in_rank; ++i) {
        padded_in[out_rank - in_rank + i] = input_shape[i];
    }

    // Compute strides
    std::vector<int64_t> in_strides(out_rank, 1);
    std::vector<int64_t> out_strides(out_rank, 1);
    for (int i = static_cast<int>(out_rank) - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * padded_in[i + 1];
        out_strides[i] = out_strides[i + 1] * target_shape[i + 1];
    }

    size_t out_total = 1;
    for (auto d : target_shape) out_total *= d;

    std::vector<float> output(out_total);

    for (size_t out_i = 0; out_i < out_total; ++out_i) {
        size_t in_idx = 0;
        size_t rem = out_i;
        for (size_t d = 0; d < out_rank; ++d) {
            int64_t coord = rem / out_strides[d];
            rem %= out_strides[d];
            // Broadcast: if input dim is 1, always use index 0
            int64_t in_coord = (padded_in[d] == 1) ? 0 : coord;
            in_idx += in_coord * in_strides[d];
        }
        output[out_i] = input[in_idx];
    }
    return output;
}

// ============================================================================
// Select reference
// output[i] = mask[i] ? input1[i] : input2[i]
// ============================================================================
inline std::vector<float> select(const std::vector<uint8_t>& mask,
                                 const std::vector<float>& input1,
                                 const std::vector<float>& input2) {
    size_t total = mask.size();
    std::vector<float> output(total);
    for (size_t i = 0; i < total; ++i) {
        output[i] = mask[i] ? input1[i] : input2[i];
    }
    return output;
}

// ============================================================================
// ScatterUpdate reference (axis=0 simplified, generic axis supported)
// data[indices[i]] = updates[i]  (along given axis)
// ============================================================================
inline std::vector<float> scatter_update(const std::vector<float>& data,
                                         const std::vector<int32_t>& indices,
                                         const std::vector<float>& updates,
                                         const std::vector<int64_t>& data_shape,
                                         const std::vector<int64_t>& idx_shape,
                                         const std::vector<int64_t>& upd_shape,
                                         int64_t axis) {
    int64_t rank = static_cast<int64_t>(data_shape.size());
    if (axis < 0) axis += rank;

    // Start with a copy of data
    std::vector<float> output = data;

    // Compute data strides
    std::vector<int64_t> data_strides(rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i)
        data_strides[i] = data_strides[i + 1] * data_shape[i + 1];

    // Compute slices before and after axis
    int64_t before = 1;
    for (int64_t i = 0; i < axis; ++i) before *= data_shape[i];
    int64_t after = 1;
    for (int64_t i = axis + 1; i < rank; ++i) after *= data_shape[i];
    int64_t num_idx = static_cast<int64_t>(indices.size());

    for (int64_t b = 0; b < before; ++b) {
        for (int64_t idx_i = 0; idx_i < num_idx; ++idx_i) {
            int32_t idx_val = indices[idx_i];
            // Normalize negative indices per OpenVINO ScatterUpdate spec
            if (idx_val < 0) idx_val += static_cast<int32_t>(data_shape[axis]);
            // Bounds check: skip out-of-range indices
            if (idx_val < 0 || idx_val >= static_cast<int32_t>(data_shape[axis])) continue;
            for (int64_t a = 0; a < after; ++a) {
                int64_t out_offset = b * data_shape[axis] * after + idx_val * after + a;
                int64_t upd_offset = b * num_idx * after + idx_i * after + a;
                output[out_offset] = updates[upd_offset];
            }
        }
    }
    return output;
}

// ============================================================================
// Depth-to-Space reference
// Input [N, C*block^2, H, W] -> Output [N, C, H*block, W*block]
// ============================================================================
inline std::vector<float> depth_to_space(const std::vector<float>& input,
                                         const std::vector<int64_t>& shape,
                                         int64_t block_size,
                                         bool blocks_first = true) {
    int64_t N = shape[0], C_in = shape[1], H = shape[2], W = shape[3];
    int64_t bs2 = block_size * block_size;
    int64_t C = C_in / bs2;
    int64_t oH = H * block_size, oW = W * block_size;
    std::vector<float> output(N * C * oH * oW);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w)
                    for (int64_t bh = 0; bh < block_size; ++bh)
                        for (int64_t bw = 0; bw < block_size; ++bw) {
                            int64_t ic;
                            if (blocks_first) ic = (bh * block_size + bw) * C + c;
                            else ic = c * bs2 + bh * block_size + bw;
                            int64_t in_idx = ((n * C_in + ic) * H + h) * W + w;
                            int64_t oh = h * block_size + bh, ow = w * block_size + bw;
                            int64_t out_idx = ((n * C + c) * oH + oh) * oW + ow;
                            output[out_idx] = input[in_idx];
                        }
    return output;
}

// ============================================================================
// Space-to-Depth reference
// Input [N, C, H, W] -> Output [N, C*block^2, H/block, W/block]
// ============================================================================
inline std::vector<float> space_to_depth(const std::vector<float>& input,
                                         const std::vector<int64_t>& shape,
                                         int64_t block_size,
                                         bool blocks_first = true) {
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int64_t bs2 = block_size * block_size;
    int64_t oC = C * bs2, oH = H / block_size, oW = W / block_size;
    std::vector<float> output(N * oC * oH * oW);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w) {
                    int64_t bh = h % block_size, bw = w % block_size;
                    int64_t oc;
                    if (blocks_first) oc = (bh * block_size + bw) * C + c;
                    else oc = c * bs2 + bh * block_size + bw;
                    int64_t oh = h / block_size, ow = w / block_size;
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    int64_t out_idx = ((n * oC + oc) * oH + oh) * oW + ow;
                    output[out_idx] = input[in_idx];
                }
    return output;
}

// ============================================================================
// ReorgYolo reference (same as space_to_depth with blocks_first)
// ============================================================================
inline std::vector<float> reorg_yolo(const std::vector<float>& input,
                                     const std::vector<int64_t>& shape,
                                     int64_t stride) {
    // GPU kernel works on flat buffer treating input as [B, C/S², H*S, W*S]
    // and output as [B, C, H, W] (same layout as input shape).
    // The framework then reinterprets [C, H, W] as [C*S², H/S, W/S].
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int64_t ic_off = C / (stride * stride);
    int64_t ih_off = H * stride;
    int64_t iw_off = W * stride;
    std::vector<float> output(N * C * H * W);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t ic = 0; ic < C; ++ic)
            for (int64_t ih = 0; ih < H; ++ih)
                for (int64_t iw = 0; iw < W; ++iw) {
                    int64_t oc = ic % ic_off;
                    int64_t offset = ic / ic_off;
                    int64_t ow = iw * stride + offset % stride;
                    int64_t oh = ih * stride + offset / stride;
                    int64_t dst = n * C * H * W + ic * H * W + ih * W + iw;
                    int64_t src = n * ic_off * ih_off * iw_off + oc * ih_off * iw_off + oh * iw_off + ow;
                    output[dst] = input[src];
                }
    return output;
}

// ============================================================================
// Shuffle Channels reference
// ============================================================================
inline std::vector<float> shuffle_channels(const std::vector<float>& input,
                                           const std::vector<int64_t>& shape,
                                           int32_t group, int32_t axis) {
    size_t rank = shape.size();
    if (axis < 0) axis += static_cast<int32_t>(rank);
    size_t total = total_elements(shape);
    std::vector<float> output(total);
    std::vector<size_t> strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * static_cast<size_t>(shape[d + 1]);
    int64_t channels_per_group = shape[axis] / group;
    for (size_t i = 0; i < total; ++i) {
        // Decompose flat index to coordinates
        std::vector<size_t> coords(rank);
        size_t tmp = i;
        for (size_t d = 0; d < rank; ++d) { coords[d] = tmp / strides[d]; tmp %= strides[d]; }
        // Shuffle: reshape [G, C/G] → transpose → [C/G, G] → flatten
        size_t old_c = coords[axis];
        size_t new_c = (old_c % channels_per_group) * group + old_c / channels_per_group;
        coords[axis] = new_c;
        size_t out_idx = 0;
        for (size_t d = 0; d < rank; ++d) out_idx += coords[d] * strides[d];
        output[out_idx] = input[i];
    }
    return output;
}

// ============================================================================
// Reverse reference (index mode: axes specify which dims to reverse)
// ============================================================================
inline std::vector<float> reverse_data(const std::vector<float>& input,
                                       const std::vector<int64_t>& shape,
                                       const std::vector<int32_t>& axes) {
    size_t rank = shape.size();
    size_t total = total_elements(shape);
    std::vector<float> output(total);
    std::vector<size_t> strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * static_cast<size_t>(shape[d + 1]);
    std::set<int32_t> axes_set(axes.begin(), axes.end());
    for (size_t i = 0; i < total; ++i) {
        std::vector<size_t> coords(rank);
        size_t tmp = i;
        for (size_t d = 0; d < rank; ++d) { coords[d] = tmp / strides[d]; tmp %= strides[d]; }
        for (auto ax : axes_set) {
            int a = ax < 0 ? ax + static_cast<int>(rank) : ax;
            coords[a] = static_cast<size_t>(shape[a]) - 1 - coords[a];
        }
        size_t out_idx = 0;
        for (size_t d = 0; d < rank; ++d) out_idx += coords[d] * strides[d];
        output[out_idx] = input[i];
    }
    return output;
}

// ============================================================================
// Roll reference (circular shift along axes)
// ============================================================================
inline std::vector<float> roll(const std::vector<float>& input,
                               const std::vector<int64_t>& shape,
                               const std::vector<int64_t>& shifts,
                               const std::vector<int64_t>& axes) {
    size_t rank = shape.size();
    size_t total = total_elements(shape);
    std::vector<float> output(total);
    std::vector<size_t> strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * static_cast<size_t>(shape[d + 1]);
    for (size_t i = 0; i < total; ++i) {
        std::vector<size_t> coords(rank);
        size_t tmp = i;
        for (size_t d = 0; d < rank; ++d) { coords[d] = tmp / strides[d]; tmp %= strides[d]; }
        std::vector<size_t> out_coords = coords;
        for (size_t a = 0; a < axes.size(); ++a) {
            int64_t ax = axes[a] < 0 ? axes[a] + static_cast<int64_t>(rank) : axes[a];
            int64_t s = shifts[a];
            int64_t dim = shape[ax];
            out_coords[ax] = static_cast<size_t>(((static_cast<int64_t>(coords[ax]) + s) % dim + dim) % dim);
        }
        size_t out_idx = 0;
        for (size_t d = 0; d < rank; ++d) out_idx += out_coords[d] * strides[d];
        output[out_idx] = input[i];
    }
    return output;
}

// ============================================================================
// Eye reference (identity-like matrix with diagonal shift)
// ============================================================================
inline std::vector<float> eye(int64_t rows, int64_t cols, int64_t shift,
                              const std::vector<int64_t>& out_shape) {
    size_t total = total_elements(out_shape);
    std::vector<float> output(total, 0.0f);
    // Last two dims are rows x cols
    size_t batch = total / (rows * cols);
    for (size_t b = 0; b < batch; ++b)
        for (int64_t r = 0; r < rows; ++r) {
            int64_t c = r + shift;
            if (c >= 0 && c < cols)
                output[b * rows * cols + r * cols + c] = 1.0f;
        }
    return output;
}

// ============================================================================
// Range reference
// ============================================================================
inline std::vector<float> range(float start, float stop, float step) {
    std::vector<float> output;
    if (step > 0) { for (float v = start; v < stop; v += step) output.push_back(v); }
    else if (step < 0) { for (float v = start; v > stop; v += step) output.push_back(v); }
    return output;
}

// ============================================================================
// CumSum reference
// ============================================================================
inline std::vector<float> cum_sum(const std::vector<float>& input,
                                  const std::vector<int64_t>& shape,
                                  int64_t axis, bool exclusive, bool reverse) {
    size_t rank = shape.size();
    if (axis < 0) axis += static_cast<int64_t>(rank);
    size_t total = total_elements(shape);
    std::vector<float> output(total);
    std::vector<size_t> strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * static_cast<size_t>(shape[d + 1]);
    size_t before = 1, after = 1;
    for (int64_t d = 0; d < axis; ++d) before *= shape[d];
    for (size_t d = axis + 1; d < rank; ++d) after *= shape[d];
    int64_t dim = shape[axis];
    for (size_t b = 0; b < before; ++b)
        for (size_t a = 0; a < after; ++a) {
            float sum = 0;
            for (int64_t i = 0; i < dim; ++i) {
                int64_t idx = reverse ? (dim - 1 - i) : i;
                size_t flat = b * dim * after + idx * after + a;
                if (exclusive) { output[flat] = sum; sum += input[flat]; }
                else { sum += input[flat]; output[flat] = sum; }
            }
        }
    return output;
}

// ============================================================================
// OneHot reference
// ============================================================================
inline std::vector<float> one_hot(const std::vector<float>& indices_f32,
                                  const std::vector<int64_t>& in_shape,
                                  int64_t depth, int64_t axis,
                                  float on_value, float off_value) {
    size_t rank_in = in_shape.size();
    if (axis < 0) axis += static_cast<int64_t>(rank_in) + 1;
    // Build output shape
    std::vector<int64_t> out_shape;
    for (size_t d = 0; d < rank_in; ++d) {
        if (static_cast<int64_t>(d) == axis) out_shape.push_back(depth);
        out_shape.push_back(in_shape[d]);
    }
    if (axis == static_cast<int64_t>(rank_in)) out_shape.push_back(depth);
    size_t total_in = total_elements(in_shape);
    size_t total_out = total_elements(out_shape);
    std::vector<float> output(total_out, off_value);
    // Compute output strides
    size_t out_rank = out_shape.size();
    std::vector<size_t> out_strides(out_rank, 1);
    for (int d = static_cast<int>(out_rank) - 2; d >= 0; --d)
        out_strides[d] = out_strides[d + 1] * static_cast<size_t>(out_shape[d + 1]);
    std::vector<size_t> in_strides(rank_in, 1);
    for (int d = static_cast<int>(rank_in) - 2; d >= 0; --d)
        in_strides[d] = in_strides[d + 1] * static_cast<size_t>(in_shape[d + 1]);
    for (size_t i = 0; i < total_in; ++i) {
        int64_t idx_val = static_cast<int64_t>(indices_f32[i]);
        if (idx_val < 0 || idx_val >= depth) continue;
        // Decompose i into input coords
        std::vector<size_t> in_coords(rank_in);
        size_t tmp = i;
        for (size_t d = 0; d < rank_in; ++d) { in_coords[d] = tmp / in_strides[d]; tmp %= in_strides[d]; }
        // Build output coords
        std::vector<size_t> out_coords;
        for (size_t d = 0; d < rank_in; ++d) {
            if (static_cast<int64_t>(d) == axis) out_coords.push_back(static_cast<size_t>(idx_val));
            out_coords.push_back(in_coords[d]);
        }
        if (axis == static_cast<int64_t>(rank_in)) out_coords.push_back(static_cast<size_t>(idx_val));
        size_t out_idx = 0;
        for (size_t d = 0; d < out_rank; ++d) out_idx += out_coords[d] * out_strides[d];
        output[out_idx] = on_value;
    }
    return output;
}

// ============================================================================
// Border (Pad) reference
// ============================================================================
inline std::vector<float> border_pad(const std::vector<float>& input,
                                     const std::vector<int64_t>& shape,
                                     const std::vector<int64_t>& pad_begin,
                                     const std::vector<int64_t>& pad_end,
                                     int pad_mode, float pad_value) {
    // pad_mode: 0=constant, 1=edge, 2=reflect, 3=symmetric
    size_t rank = shape.size();
    std::vector<int64_t> out_shape(rank);
    for (size_t d = 0; d < rank; ++d)
        out_shape[d] = shape[d] + (d < pad_begin.size() ? pad_begin[d] : 0) + (d < pad_end.size() ? pad_end[d] : 0);
    size_t total = total_elements(out_shape);
    std::vector<float> output(total, pad_value);
    std::vector<size_t> in_strides(rank, 1), out_strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * static_cast<size_t>(shape[d + 1]);
        out_strides[d] = out_strides[d + 1] * static_cast<size_t>(out_shape[d + 1]);
    }
    for (size_t i = 0; i < total; ++i) {
        std::vector<size_t> out_coords(rank);
        size_t tmp = i;
        for (size_t d = 0; d < rank; ++d) { out_coords[d] = tmp / out_strides[d]; tmp %= out_strides[d]; }
        bool valid = true;
        std::vector<size_t> in_coords(rank);
        for (size_t d = 0; d < rank; ++d) {
            int64_t pb = d < pad_begin.size() ? pad_begin[d] : 0;
            int64_t c = static_cast<int64_t>(out_coords[d]) - pb;
            if (pad_mode == 0) { // constant
                if (c < 0 || c >= shape[d]) { valid = false; break; }
                in_coords[d] = c;
            } else if (pad_mode == 1) { // edge
                in_coords[d] = std::max(int64_t(0), std::min(c, shape[d] - 1));
            } else if (pad_mode == 2) { // reflect
                if (c < 0) c = -c;
                if (c >= shape[d]) c = 2 * (shape[d] - 1) - c;
                c = std::max(int64_t(0), std::min(c, shape[d] - 1));
                in_coords[d] = c;
            } else { // symmetric
                if (c < 0) c = -c - 1;
                if (c >= shape[d]) c = 2 * shape[d] - 1 - c;
                c = std::max(int64_t(0), std::min(c, shape[d] - 1));
                in_coords[d] = c;
            }
        }
        if (!valid) continue;
        size_t in_idx = 0;
        for (size_t d = 0; d < rank; ++d) in_idx += in_coords[d] * in_strides[d];
        output[i] = input[in_idx];
    }
    return output;
}

// ============================================================================
// Slice reference (like strided_slice but with start/stop/step/axes)
// ============================================================================
inline std::vector<float> slice_ref(const std::vector<float>& input,
                                    const std::vector<int64_t>& shape,
                                    const std::vector<int64_t>& start,
                                    const std::vector<int64_t>& stop,
                                    const std::vector<int64_t>& step,
                                    const std::vector<int64_t>& axes) {
    size_t rank = shape.size();
    // Full-dim start/stop/step
    std::vector<int64_t> full_start(rank, 0), full_stop(shape), full_step(rank, 1);
    for (size_t i = 0; i < axes.size(); ++i) {
        int64_t ax = axes[i] < 0 ? axes[i] + static_cast<int64_t>(rank) : axes[i];
        full_start[ax] = start[i];
        full_stop[ax] = stop[i];
        full_step[ax] = step[i];
    }
    // Normalize and compute output shape
    std::vector<int64_t> out_shape(rank);
    for (size_t d = 0; d < rank; ++d) {
        int64_t s = full_start[d], e = full_stop[d], st = full_step[d];
        if (s < 0) s += shape[d];
        if (e < 0) e += shape[d];
        s = std::max(int64_t(0), std::min(s, shape[d]));
        e = std::max(int64_t(0), std::min(e, shape[d]));
        if (st > 0) out_shape[d] = std::max(int64_t(0), (e - s + st - 1) / st);
        else out_shape[d] = std::max(int64_t(0), (s - e + (-st) - 1) / (-st));
        full_start[d] = s; full_stop[d] = e; full_step[d] = st;
    }
    size_t total_out = total_elements(out_shape);
    std::vector<float> output(total_out);
    std::vector<size_t> in_strides(rank, 1), out_strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * static_cast<size_t>(shape[d + 1]);
        out_strides[d] = out_strides[d + 1] * static_cast<size_t>(out_shape[d + 1]);
    }
    for (size_t i = 0; i < total_out; ++i) {
        std::vector<size_t> out_coords(rank);
        size_t tmp = i;
        for (size_t d = 0; d < rank; ++d) { out_coords[d] = tmp / out_strides[d]; tmp %= out_strides[d]; }
        size_t in_idx = 0;
        for (size_t d = 0; d < rank; ++d) {
            int64_t in_d = full_start[d] + static_cast<int64_t>(out_coords[d]) * full_step[d];
            in_idx += static_cast<size_t>(in_d) * in_strides[d];
        }
        output[i] = input[in_idx];
    }
    return output;
}

// ============================================================================
// LRN reference
// ============================================================================
inline std::vector<float> lrn(const std::vector<float>& input,
                              const std::vector<int64_t>& shape,
                              int32_t size, float k, float alpha, float beta,
                              bool across_channels) {
    int64_t N = shape[0], C = shape[1], H = shape.size() > 2 ? shape[2] : 1, W = shape.size() > 3 ? shape[3] : 1;
    size_t total = N * C * H * W;
    std::vector<float> output(total);
    int half = size / 2;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w) {
                    float sum_sq = 0;
                    if (across_channels) {
                        for (int64_t i = std::max(int64_t(0), c - half); i <= std::min(C - 1, c + half); ++i)
                            sum_sq += input[((n * C + i) * H + h) * W + w] * input[((n * C + i) * H + h) * W + w];
                    } else {
                        for (int64_t dh = -half; dh <= half; ++dh)
                            for (int64_t dw = -half; dw <= half; ++dw) {
                                int64_t nh = h + dh, nw = w + dw;
                                if (nh >= 0 && nh < H && nw >= 0 && nw < W)
                                    sum_sq += input[((n * C + c) * H + nh) * W + nw] * input[((n * C + c) * H + nh) * W + nw];
                            }
                    }
                    float val = input[((n * C + c) * H + h) * W + w];
                    output[((n * C + c) * H + h) * W + w] = val / std::pow(k + alpha / size * sum_sq, beta);
                }
    return output;
}

// ============================================================================
// GRN reference
// ============================================================================
inline std::vector<float> grn(const std::vector<float>& input,
                              const std::vector<int64_t>& shape,
                              float bias) {
    int64_t N = shape[0], C = shape[1], H = shape.size() > 2 ? shape[2] : 1, W = shape.size() > 3 ? shape[3] : 1;
    size_t total = N * C * H * W;
    std::vector<float> output(total);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w) {
                float sum_sq = 0;
                for (int64_t c = 0; c < C; ++c)
                    sum_sq += input[((n * C + c) * H + h) * W + w] * input[((n * C + c) * H + h) * W + w];
                float norm = std::sqrt(sum_sq + bias);
                for (int64_t c = 0; c < C; ++c)
                    output[((n * C + c) * H + h) * W + w] = input[((n * C + c) * H + h) * W + w] / norm;
            }
    return output;
}

// ============================================================================
// Batch-to-Space reference
// ============================================================================
inline std::vector<float> batch_to_space(const std::vector<float>& input,
                                         const std::vector<int64_t>& shape,
                                         const std::vector<int64_t>& block_shape,
                                         const std::vector<int64_t>& crops_begin,
                                         const std::vector<int64_t>& crops_end) {
    // shape = [N, C, D..., H, W], block_shape for each spatial + batch dim
    size_t rank = shape.size();
    // Compute output shape
    int64_t block_prod = 1;
    for (size_t d = 1; d < rank; ++d) block_prod *= (d < block_shape.size() ? block_shape[d] : 1);
    std::vector<int64_t> out_shape(rank);
    out_shape[0] = shape[0] / block_prod;
    for (size_t d = 1; d < rank; ++d) {
        int64_t bs = d < block_shape.size() ? block_shape[d] : 1;
        int64_t cb = d < crops_begin.size() ? crops_begin[d] : 0;
        int64_t ce = d < crops_end.size() ? crops_end[d] : 0;
        out_shape[d] = shape[d] * bs - cb - ce;
    }
    size_t total_out = total_elements(out_shape);
    size_t total_in = total_elements(shape);
    std::vector<float> output(total_out, 0.0f);
    // Iterate over input and map to output
    std::vector<size_t> in_strides(rank, 1), out_strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * static_cast<size_t>(shape[d + 1]);
        out_strides[d] = out_strides[d + 1] * static_cast<size_t>(out_shape[d + 1]);
    }
    for (size_t i = 0; i < total_in; ++i) {
        std::vector<size_t> in_coords(rank);
        size_t tmp = i;
        for (size_t d = 0; d < rank; ++d) { in_coords[d] = tmp / in_strides[d]; tmp %= in_strides[d]; }
        // Decompose batch index into spatial block indices
        int64_t batch_idx = in_coords[0];
        int64_t out_batch = batch_idx;
        std::vector<int64_t> block_indices(rank, 0);
        for (int d = static_cast<int>(rank) - 1; d >= 1; --d) {
            int64_t bs = d < static_cast<int>(block_shape.size()) ? block_shape[d] : 1;
            block_indices[d] = out_batch % bs;
            out_batch /= bs;
        }
        // Map to output coordinates
        std::vector<int64_t> out_coords(rank);
        out_coords[0] = out_batch;
        bool valid = true;
        for (size_t d = 1; d < rank; ++d) {
            int64_t bs = d < block_shape.size() ? block_shape[d] : 1;
            int64_t cb = d < crops_begin.size() ? crops_begin[d] : 0;
            int64_t oc = static_cast<int64_t>(in_coords[d]) * bs + block_indices[d] - cb;
            if (oc < 0 || oc >= out_shape[d]) { valid = false; break; }
            out_coords[d] = oc;
        }
        if (!valid) continue;
        size_t out_idx = 0;
        for (size_t d = 0; d < rank; ++d) out_idx += out_coords[d] * out_strides[d];
        output[out_idx] = input[i];
    }
    return output;
}

// ============================================================================
// Space-to-Batch reference
// ============================================================================
inline std::vector<float> space_to_batch(const std::vector<float>& input,
                                         const std::vector<int64_t>& shape,
                                         const std::vector<int64_t>& block_shape,
                                         const std::vector<int64_t>& pads_begin,
                                         const std::vector<int64_t>& pads_end) {
    size_t rank = shape.size();
    int64_t block_prod = 1;
    for (size_t d = 1; d < rank; ++d) block_prod *= (d < block_shape.size() ? block_shape[d] : 1);
    std::vector<int64_t> out_shape(rank);
    out_shape[0] = shape[0] * block_prod;
    for (size_t d = 1; d < rank; ++d) {
        int64_t bs = d < block_shape.size() ? block_shape[d] : 1;
        int64_t pb = d < pads_begin.size() ? pads_begin[d] : 0;
        int64_t pe = d < pads_end.size() ? pads_end[d] : 0;
        out_shape[d] = (shape[d] + pb + pe) / bs;
    }
    size_t total_out = total_elements(out_shape);
    std::vector<float> output(total_out, 0.0f);
    std::vector<size_t> in_strides(rank, 1), out_strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * static_cast<size_t>(shape[d + 1]);
        out_strides[d] = out_strides[d + 1] * static_cast<size_t>(out_shape[d + 1]);
    }
    for (size_t i = 0; i < total_out; ++i) {
        std::vector<size_t> out_coords(rank);
        size_t tmp = i;
        for (size_t d = 0; d < rank; ++d) { out_coords[d] = tmp / out_strides[d]; tmp %= out_strides[d]; }
        int64_t batch_idx = out_coords[0];
        int64_t in_batch = batch_idx;
        std::vector<int64_t> block_indices(rank, 0);
        for (int d = static_cast<int>(rank) - 1; d >= 1; --d) {
            int64_t bs = d < static_cast<int>(block_shape.size()) ? block_shape[d] : 1;
            block_indices[d] = in_batch % bs;
            in_batch /= bs;
        }
        bool valid = true;
        std::vector<int64_t> in_coords(rank);
        in_coords[0] = in_batch;
        for (size_t d = 1; d < rank; ++d) {
            int64_t bs = d < block_shape.size() ? block_shape[d] : 1;
            int64_t pb = d < pads_begin.size() ? pads_begin[d] : 0;
            int64_t ic = static_cast<int64_t>(out_coords[d]) * bs + block_indices[d] - pb;
            if (ic < 0 || ic >= shape[d]) { valid = false; break; }
            in_coords[d] = ic;
        }
        if (!valid) continue;
        size_t in_idx = 0;
        for (size_t d = 0; d < rank; ++d) in_idx += in_coords[d] * in_strides[d];
        output[i] = input[in_idx];
    }
    return output;
}

// ============================================================================
// EmbeddingBag (packed_sum) reference
// ============================================================================
inline std::vector<float> embedding_bag_packed(const std::vector<float>& table,
                                               const std::vector<int64_t>& table_shape,
                                               const std::vector<float>& indices_f32,
                                               const std::vector<int64_t>& idx_shape) {
    int64_t V = table_shape[0], D = table_shape[1];
    int64_t num_bags = idx_shape[0], bag_size = idx_shape[1];
    std::vector<float> output(num_bags * D, 0.0f);
    for (int64_t b = 0; b < num_bags; ++b)
        for (int64_t j = 0; j < bag_size; ++j) {
            int64_t idx = static_cast<int64_t>(indices_f32[b * bag_size + j]);
            if (idx < 0 || idx >= V) continue;
            for (int64_t d = 0; d < D; ++d)
                output[b * D + d] += table[idx * D + d];
        }
    return output;
}

// ============================================================================
// GatherND reference
// ============================================================================
inline std::vector<float> gather_nd(const std::vector<float>& data,
                                    const std::vector<float>& indices_f32,
                                    const std::vector<int64_t>& data_shape,
                                    const std::vector<int64_t>& idx_shape,
                                    int64_t batch_dims) {
    size_t data_rank = data_shape.size();
    size_t idx_rank = idx_shape.size();
    int64_t last_idx_dim = idx_shape[idx_rank - 1];
    // Compute data strides
    std::vector<size_t> data_strides(data_rank, 1);
    for (int d = static_cast<int>(data_rank) - 2; d >= 0; --d)
        data_strides[d] = data_strides[d + 1] * static_cast<size_t>(data_shape[d + 1]);
    // Batch size
    size_t batch_size = 1;
    for (int64_t d = 0; d < batch_dims; ++d) batch_size *= data_shape[d];
    // Number of index tuples
    size_t num_idx = 1;
    for (size_t d = 0; d < idx_rank - 1; ++d) num_idx *= idx_shape[d];
    size_t idx_per_batch = num_idx / batch_size;
    // Slice size after gathered dims
    size_t slice_size = 1;
    for (size_t d = batch_dims + last_idx_dim; d < data_rank; ++d) slice_size *= data_shape[d];
    size_t total_out = num_idx * slice_size;
    std::vector<float> output(total_out);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < idx_per_batch; ++i) {
            size_t idx_flat = b * idx_per_batch + i;
            // Compute data offset from batch + gathered indices
            size_t data_offset = 0;
            for (int64_t d = 0; d < batch_dims; ++d) {
                size_t coord = b;
                for (int64_t dd = d + 1; dd < batch_dims; ++dd) coord /= data_shape[dd];
                coord %= data_shape[d];
                data_offset += coord * data_strides[d];
            }
            for (int64_t j = 0; j < last_idx_dim; ++j) {
                int64_t idx_val = static_cast<int64_t>(indices_f32[idx_flat * last_idx_dim + j]);
                int64_t dim_size = data_shape[batch_dims + j];
                if (idx_val < 0) idx_val += dim_size;
                idx_val = std::max(int64_t(0), std::min(idx_val, dim_size - 1));
                data_offset += idx_val * data_strides[batch_dims + j];
            }
            for (size_t s = 0; s < slice_size; ++s) {
                output[idx_flat * slice_size + s] = data[data_offset + s];
            }
        }
    }
    return output;
}

// ============================================================================
// GatherTree reference (beam search traceback)
// ============================================================================
inline std::vector<float> gather_tree(const std::vector<float>& step_ids,
                                      const std::vector<float>& parent_ids,
                                      const std::vector<float>& max_seq_len,
                                      const std::vector<float>& end_token,
                                      const std::vector<int64_t>& shape) {
    int64_t T = shape[0], B = shape[1], beam = shape[2];
    float end_tok = end_token[0];
    std::vector<float> output(T * B * beam, end_tok);
    for (int64_t b = 0; b < B; ++b) {
        int64_t max_t = std::min(T, static_cast<int64_t>(max_seq_len[b]));
        for (int64_t k = 0; k < beam; ++k) {
            output[((max_t > 0 ? max_t - 1 : 0) * B + b) * beam + k] = step_ids[((max_t > 0 ? max_t - 1 : 0) * B + b) * beam + k];
            int64_t parent = static_cast<int64_t>(parent_ids[((max_t > 0 ? max_t - 1 : 0) * B + b) * beam + k]);
            for (int64_t t = max_t - 2; t >= 0; --t) {
                output[(t * B + b) * beam + k] = step_ids[(t * B + b) * beam + parent];
                parent = static_cast<int64_t>(parent_ids[(t * B + b) * beam + parent]);
            }
            // Replace values after first end_token
            bool found_end = false;
            for (int64_t t = 0; t < max_t; ++t) {
                if (found_end) output[(t * B + b) * beam + k] = end_tok;
                else if (output[(t * B + b) * beam + k] == end_tok) found_end = true;
            }
        }
    }
    return output;
}

// ============================================================================
// Bucketize reference
// ============================================================================
inline std::vector<float> bucketize(const std::vector<float>& input,
                                    const std::vector<float>& boundaries,
                                    bool right_mode) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float val = input[i];
        int64_t lo = 0, hi = static_cast<int64_t>(boundaries.size());
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            bool cmp = right_mode ? (boundaries[mid] <= val) : (boundaries[mid] < val);
            if (cmp) lo = mid + 1; else hi = mid;
        }
        output[i] = static_cast<float>(lo);
    }
    return output;
}

// ============================================================================
// SearchSorted reference
// ============================================================================
inline std::vector<float> search_sorted(const std::vector<float>& sorted,
                                        const std::vector<float>& values,
                                        const std::vector<int64_t>& sorted_shape,
                                        const std::vector<int64_t>& values_shape,
                                        bool right_mode) {
    // sorted: [..., L], values: [..., N] → output: [..., N]
    size_t L = sorted_shape.back();
    size_t N = values_shape.back();
    size_t batch = 1;
    for (size_t d = 0; d + 1 < sorted_shape.size(); ++d) batch *= sorted_shape[d];
    std::vector<float> output(batch * N);
    for (size_t b = 0; b < batch; ++b) {
        const float* s = sorted.data() + b * L;
        for (size_t i = 0; i < N; ++i) {
            float val = values[b * N + i];
            int64_t lo = 0, hi = static_cast<int64_t>(L);
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                bool cmp = right_mode ? (s[mid] <= val) : (s[mid] < val);
                if (cmp) lo = mid + 1; else hi = mid;
            }
            output[b * N + i] = static_cast<float>(lo);
        }
    }
    return output;
}

// ============================================================================
// SegmentMax reference
// ============================================================================
inline std::vector<float> segment_max(const std::vector<float>& data,
                                      const std::vector<float>& seg_ids_f32,
                                      const std::vector<int64_t>& data_shape,
                                      int64_t num_segments) {
    int64_t N = data_shape[0];
    size_t feat_size = 1;
    for (size_t d = 1; d < data_shape.size(); ++d) feat_size *= data_shape[d];
    std::vector<float> output(num_segments * feat_size, 0.0f);
    std::vector<bool> initialized(num_segments, false);
    for (int64_t i = 0; i < N; ++i) {
        int64_t seg = static_cast<int64_t>(seg_ids_f32[i]);
        if (seg < 0 || seg >= num_segments) continue;
        if (!initialized[seg]) {
            for (size_t f = 0; f < feat_size; ++f)
                output[seg * feat_size + f] = data[i * feat_size + f];
            initialized[seg] = true;
        } else {
            for (size_t f = 0; f < feat_size; ++f)
                output[seg * feat_size + f] = std::max(output[seg * feat_size + f], data[i * feat_size + f]);
        }
    }
    return output;
}

// ============================================================================
// ExtractImagePatches reference
// ============================================================================
inline std::vector<float> extract_image_patches(const std::vector<float>& input,
                                                const std::vector<int64_t>& shape,
                                                const std::vector<int64_t>& sizes,
                                                const std::vector<int64_t>& strides,
                                                const std::vector<int64_t>& rates) {
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int64_t kH = sizes[0], kW = sizes[1];
    int64_t sH = strides[0], sW = strides[1];
    int64_t rH = rates[0], rW = rates[1];
    int64_t oH = (H - (kH - 1) * rH - 1) / sH + 1;
    int64_t oW = (W - (kW - 1) * rW - 1) / sW + 1;
    int64_t oC = C * kH * kW;
    std::vector<float> output(N * oC * oH * oW, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t oh = 0; oh < oH; ++oh)
            for (int64_t ow = 0; ow < oW; ++ow)
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t ph = 0; ph < kH; ++ph)
                        for (int64_t pw = 0; pw < kW; ++pw) {
                            int64_t ih = oh * sH + ph * rH;
                            int64_t iw = ow * sW + pw * rW;
                            int64_t oc = (c * kH + ph) * kW + pw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                output[((n * oC + oc) * oH + oh) * oW + ow] = input[((n * C + c) * H + ih) * W + iw];
                        }
    return output;
}

// ============================================================================
// ReverseSequence reference
// ============================================================================
inline std::vector<float> reverse_sequence(const std::vector<float>& data,
                                           const std::vector<float>& seq_lens_f32,
                                           const std::vector<int64_t>& shape,
                                           int64_t seq_axis, int64_t batch_axis) {
    size_t rank = shape.size();
    size_t total = total_elements(shape);
    std::vector<float> output(data);
    std::vector<size_t> strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * shape[d + 1];
    // Iterate over all elements, reverse seq_axis for each batch element
    for (size_t i = 0; i < total; ++i) {
        std::vector<size_t> coords(rank);
        size_t tmp = i;
        for (size_t d = 0; d < rank; ++d) { coords[d] = tmp / strides[d]; tmp %= strides[d]; }
        int64_t batch_idx = coords[batch_axis];
        int64_t seq_idx = coords[seq_axis];
        int64_t seq_len = static_cast<int64_t>(seq_lens_f32[batch_idx]);
        if (seq_idx < seq_len) {
            auto new_coords = coords;
            new_coords[seq_axis] = seq_len - 1 - seq_idx;
            size_t src_idx = 0;
            for (size_t d = 0; d < rank; ++d) src_idx += new_coords[d] * strides[d];
            output[i] = data[src_idx];
        }
    }
    return output;
}

// ============================================================================
// FakeConvert reference (round-trip quantize to f8e4m3)
// ============================================================================
inline std::vector<float> fake_convert(const std::vector<float>& input,
                                       const std::vector<float>& scale,
                                       const std::vector<int64_t>& input_shape,
                                       const std::vector<int64_t>& scale_shape) {
    size_t total = input.size();
    // Broadcast scale to input shape
    size_t scale_total = scale.size();
    std::vector<float> output(total);
    // f8e4m3 range: +-448, 4 bit exponent, 3 bit mantissa
    float max_val = 448.0f;
    for (size_t i = 0; i < total; ++i) {
        float s = scale[i % scale_total];
        float scaled = input[i] / s;
        scaled = std::max(-max_val, std::min(max_val, scaled));
        // Round to f8e4m3 precision (3 mantissa bits → round to nearest 1/8)
        // Simplified: just round to reduce bits
        int exp;
        float mant = std::frexp(std::abs(scaled), &exp);
        mant = std::round(mant * 8.0f) / 8.0f;
        float rounded = std::ldexp(mant, exp) * (scaled < 0 ? -1.0f : 1.0f);
        output[i] = rounded * s;
    }
    return output;
}

// ============================================================================
// AdaptivePooling reference
// ============================================================================
inline std::vector<float> adaptive_pooling(const std::vector<float>& input,
                                           const std::vector<int64_t>& shape,
                                           int64_t out_h, int64_t out_w,
                                           bool max_mode) {
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    std::vector<float> output(N * C * out_h * out_w);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t oh = 0; oh < out_h; ++oh)
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    int64_t h_start = oh * H / out_h;
                    int64_t h_end = (oh + 1) * H / out_h;
                    int64_t w_start = ow * W / out_w;
                    int64_t w_end = (ow + 1) * W / out_w;
                    float val = max_mode ? -std::numeric_limits<float>::infinity() : 0.0f;
                    int count = 0;
                    for (int64_t h = h_start; h < h_end; ++h)
                        for (int64_t w = w_start; w < w_end; ++w) {
                            float v = input[((n * C + c) * H + h) * W + w];
                            if (max_mode) val = std::max(val, v);
                            else val += v;
                            count++;
                        }
                    if (!max_mode && count > 0) val /= count;
                    output[((n * C + c) * out_h + oh) * out_w + ow] = val;
                }
    return output;
}

// ============================================================================
// ArgMaxMin reference (returns indices as float)
// ============================================================================
inline std::vector<float> arg_max_min(const std::vector<float>& input,
                                      const std::vector<int64_t>& shape,
                                      int64_t axis, int64_t top_k, bool max_mode) {
    size_t rank = shape.size();
    if (axis < 0) axis += static_cast<int64_t>(rank);
    size_t before = 1, after = 1;
    for (int64_t d = 0; d < axis; ++d) before *= shape[d];
    for (size_t d = axis + 1; d < rank; ++d) after *= shape[d];
    int64_t dim = shape[axis];
    int64_t k = std::min(top_k, dim);
    std::vector<float> output(before * k * after);
    for (size_t b = 0; b < before; ++b)
        for (size_t a = 0; a < after; ++a) {
            std::vector<std::pair<float, int64_t>> vals(dim);
            for (int64_t d = 0; d < dim; ++d)
                vals[d] = {input[(b * dim + d) * after + a], d};
            if (max_mode)
                std::partial_sort(vals.begin(), vals.begin() + k, vals.end(),
                    [](auto& a, auto& b) { return a.first > b.first; });
            else
                std::partial_sort(vals.begin(), vals.begin() + k, vals.end(),
                    [](auto& a, auto& b) { return a.first < b.first; });
            for (int64_t j = 0; j < k; ++j)
                output[(b * k + j) * after + a] = static_cast<float>(vals[j].second);
        }
    return output;
}

// ============================================================================
// Col2Im reference
// ============================================================================
inline std::vector<float> col2im(const std::vector<float>& input,
                                 const std::vector<int64_t>& in_shape,
                                 const std::vector<int64_t>& output_size,
                                 const std::vector<int64_t>& kernel_shape,
                                 const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& dilations,
                                 const std::vector<int64_t>& pads_begin) {
    int64_t N = in_shape[0], L = in_shape.size() > 2 ? in_shape[2] : in_shape[1];
    int64_t oH = output_size[0], oW = output_size[1];
    int64_t kH = kernel_shape[0], kW = kernel_shape[1];
    int64_t sH = strides.size() > 0 ? strides[0] : 1, sW = strides.size() > 1 ? strides[1] : 1;
    int64_t dH = dilations.size() > 0 ? dilations[0] : 1, dW = dilations.size() > 1 ? dilations[1] : 1;
    int64_t pH = pads_begin.size() > 0 ? pads_begin[0] : 0, pW = pads_begin.size() > 1 ? pads_begin[1] : 0;
    int64_t C = in_shape[1] / (kH * kW);
    std::vector<float> output(N * C * oH * oW, 0.0f);
    int64_t outH = (oH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t outW = (oW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t ph = 0; ph < kH; ++ph)
                for (int64_t pw = 0; pw < kW; ++pw) {
                    int64_t col_c = (c * kH + ph) * kW + pw;
                    for (int64_t oh = 0; oh < outH; ++oh)
                        for (int64_t ow = 0; ow < outW; ++ow) {
                            int64_t ih = oh * sH + ph * dH - pH;
                            int64_t iw = ow * sW + pw * dW - pW;
                            if (ih >= 0 && ih < oH && iw >= 0 && iw < oW) {
                                int64_t col_idx = oh * outW + ow;
                                output[((n * C + c) * oH + ih) * oW + iw] += input[(n * in_shape[1] + col_c) * L + col_idx];
                            }
                        }
                }
    return output;
}

// ============================================================================
// Grid Sample reference (bilinear, nearest, bicubic)
// ============================================================================
inline std::vector<float> grid_sample(const std::vector<float>& data,
                                      const std::vector<float>& grid,
                                      const std::vector<int64_t>& data_shape,
                                      const std::vector<int64_t>& grid_shape,
                                      int grid_mode, int padding_mode, bool align_corners) {
    int64_t N = data_shape[0], C = data_shape[1], H = data_shape[2], W = data_shape[3];
    int64_t oH = grid_shape[1], oW = grid_shape[2];
    std::vector<float> output(N * C * oH * oW, 0.0f);
    auto unnorm = [&](float x, int64_t size) -> float {
        if (align_corners) return (x + 1.0f) / 2.0f * (size - 1);
        else return ((x + 1.0f) * size - 1.0f) / 2.0f;
    };
    auto sample = [&](float y, float x, int64_t n, int64_t c) -> float {
        auto clamp_coord = [&](int64_t coord, int64_t size) -> int64_t {
            return std::max(int64_t(0), std::min(coord, size - 1));
        };
        auto get_pixel = [&](int64_t iy, int64_t ix) -> float {
            if (padding_mode == 0) { // zeros
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) return 0.0f;
            } else if (padding_mode == 1) { // border
                iy = clamp_coord(iy, H); ix = clamp_coord(ix, W);
            } else { // reflection
                auto reflect = [](int64_t c, int64_t s) -> int64_t {
                    if (s <= 1) return 0;
                    while (c < 0 || c >= s) { if (c < 0) c = -c; if (c >= s) c = 2 * (s - 1) - c; }
                    return c;
                };
                iy = reflect(iy, H); ix = reflect(ix, W);
            }
            return data[((n * C + c) * H + iy) * W + ix];
        };
        if (grid_mode == 2) { // nearest
            return get_pixel(static_cast<int64_t>(std::round(y)), static_cast<int64_t>(std::round(x)));
        }
        // bilinear
        int64_t y0 = static_cast<int64_t>(std::floor(y)), x0 = static_cast<int64_t>(std::floor(x));
        float fy = y - y0, fx = x - x0;
        return get_pixel(y0, x0) * (1 - fy) * (1 - fx) +
               get_pixel(y0, x0 + 1) * (1 - fy) * fx +
               get_pixel(y0 + 1, x0) * fy * (1 - fx) +
               get_pixel(y0 + 1, x0 + 1) * fy * fx;
    };
    for (int64_t n = 0; n < N; ++n)
        for (int64_t oh = 0; oh < oH; ++oh)
            for (int64_t ow = 0; ow < oW; ++ow) {
                float gx = grid[((n * oH + oh) * oW + ow) * 2 + 0];
                float gy = grid[((n * oH + oh) * oW + ow) * 2 + 1];
                float ix = unnorm(gx, W), iy = unnorm(gy, H);
                for (int64_t c = 0; c < C; ++c)
                    output[((n * C + c) * oH + oh) * oW + ow] = sample(iy, ix, n, c);
            }
    return output;
}

// ============================================================================
// ROI Pooling reference (max mode)
// ============================================================================
inline std::vector<float> roi_pooling(const std::vector<float>& feat,
                                      const std::vector<float>& rois,
                                      const std::vector<int64_t>& feat_shape,
                                      const std::vector<int64_t>& rois_shape,
                                      int64_t pooled_h, int64_t pooled_w,
                                      float spatial_scale) {
    int64_t N = feat_shape[0], C = feat_shape[1], H = feat_shape[2], W = feat_shape[3];
    int64_t num_rois = rois_shape[0];
    std::vector<float> output(num_rois * C * pooled_h * pooled_w, 0.0f);
    for (int64_t r = 0; r < num_rois; ++r) {
        int64_t batch_idx = static_cast<int64_t>(rois[r * 5 + 0]);
        batch_idx = std::max(int64_t(0), std::min(batch_idx, N - 1));
        float x1 = rois[r * 5 + 1] * spatial_scale, y1 = rois[r * 5 + 2] * spatial_scale;
        float x2 = rois[r * 5 + 3] * spatial_scale, y2 = rois[r * 5 + 4] * spatial_scale;
        float roi_h = std::max(y2 - y1, 1.0f), roi_w = std::max(x2 - x1, 1.0f);
        float bin_h = roi_h / pooled_h, bin_w = roi_w / pooled_w;
        for (int64_t c = 0; c < C; ++c)
            for (int64_t ph = 0; ph < pooled_h; ++ph)
                for (int64_t pw = 0; pw < pooled_w; ++pw) {
                    int64_t hstart = static_cast<int64_t>(std::floor(y1 + ph * bin_h));
                    int64_t hend = static_cast<int64_t>(std::ceil(y1 + (ph + 1) * bin_h));
                    int64_t wstart = static_cast<int64_t>(std::floor(x1 + pw * bin_w));
                    int64_t wend = static_cast<int64_t>(std::ceil(x1 + (pw + 1) * bin_w));
                    hstart = std::max(int64_t(0), hstart); hend = std::min(H, hend);
                    wstart = std::max(int64_t(0), wstart); wend = std::min(W, wend);
                    float maxval = -std::numeric_limits<float>::infinity();
                    for (int64_t h = hstart; h < hend; ++h)
                        for (int64_t w = wstart; w < wend; ++w)
                            maxval = std::max(maxval, feat[((batch_idx * C + c) * H + h) * W + w]);
                    if (maxval == -std::numeric_limits<float>::infinity()) maxval = 0;
                    output[((r * C + c) * pooled_h + ph) * pooled_w + pw] = maxval;
                }
    }
    return output;
}

// ============================================================================
// ROI Align reference (avg mode)
// ============================================================================
inline std::vector<float> roi_align(const std::vector<float>& feat,
                                    const std::vector<float>& rois,
                                    const std::vector<float>& batch_idx_f32,
                                    const std::vector<int64_t>& feat_shape,
                                    int64_t pooled_h, int64_t pooled_w,
                                    int64_t sampling_ratio, float spatial_scale,
                                    bool avg_mode) {
    int64_t C = feat_shape[1], H = feat_shape[2], W = feat_shape[3];
    int64_t num_rois = static_cast<int64_t>(rois.size()) / 4;
    std::vector<float> output(num_rois * C * pooled_h * pooled_w, 0.0f);
    auto bilinear = [&](int64_t n, int64_t c, float y, float x) -> float {
        if (y < -1.0f || y > H || x < -1.0f || x > W) return 0.0f;
        y = std::max(0.0f, y); x = std::max(0.0f, x);
        int64_t y0 = static_cast<int64_t>(y), x0 = static_cast<int64_t>(x);
        int64_t y1 = std::min(y0 + 1, H - 1), x1 = std::min(x0 + 1, W - 1);
        y0 = std::min(y0, H - 1); x0 = std::min(x0, W - 1);
        float ly = y - y0, lx = x - x0;
        return feat[((n * C + c) * H + y0) * W + x0] * (1 - ly) * (1 - lx) +
               feat[((n * C + c) * H + y0) * W + x1] * (1 - ly) * lx +
               feat[((n * C + c) * H + y1) * W + x0] * ly * (1 - lx) +
               feat[((n * C + c) * H + y1) * W + x1] * ly * lx;
    };
    for (int64_t r = 0; r < num_rois; ++r) {
        int64_t n = static_cast<int64_t>(batch_idx_f32[r]);
        float x1 = rois[r * 4] * spatial_scale, y1 = rois[r * 4 + 1] * spatial_scale;
        float x2 = rois[r * 4 + 2] * spatial_scale, y2 = rois[r * 4 + 3] * spatial_scale;
        float roi_h = y2 - y1, roi_w = x2 - x1;
        float bin_h = roi_h / pooled_h, bin_w = roi_w / pooled_w;
        int64_t sr_h = sampling_ratio > 0 ? sampling_ratio : static_cast<int64_t>(std::ceil(bin_h));
        int64_t sr_w = sampling_ratio > 0 ? sampling_ratio : static_cast<int64_t>(std::ceil(bin_w));
        for (int64_t c = 0; c < C; ++c)
            for (int64_t ph = 0; ph < pooled_h; ++ph)
                for (int64_t pw = 0; pw < pooled_w; ++pw) {
                    float val = avg_mode ? 0.0f : -std::numeric_limits<float>::infinity();
                    for (int64_t sy = 0; sy < sr_h; ++sy)
                        for (int64_t sx = 0; sx < sr_w; ++sx) {
                            float fy = y1 + ph * bin_h + (sy + 0.5f) * bin_h / sr_h;
                            float fx = x1 + pw * bin_w + (sx + 0.5f) * bin_w / sr_w;
                            float v = bilinear(n, c, fy, fx);
                            if (avg_mode) val += v; else val = std::max(val, v);
                        }
                    if (avg_mode) val /= (sr_h * sr_w);
                    output[((r * C + c) * pooled_h + ph) * pooled_w + pw] = val;
                }
    }
    return output;
}

// ============================================================================
// Deconvolution (Transposed Convolution) reference
// ============================================================================
inline std::vector<float> deconv2d(const std::vector<float>& input,
                                   const std::vector<float>& weights,
                                   const std::vector<int64_t>& in_shape,
                                   const std::vector<int64_t>& w_shape,
                                   const std::vector<int64_t>& strides,
                                   const std::vector<int64_t>& dilations,
                                   const std::vector<int64_t>& pads_begin,
                                   const std::vector<int64_t>& pads_end,
                                   int64_t groups) {
    int64_t N = in_shape[0], IC = in_shape[1];
    int64_t iH = in_shape[2], iW = in_shape[3];
    int64_t OC = w_shape[1] * groups; // w_shape: [IC, OC/groups, kH, kW]
    int64_t kH = w_shape[2], kW = w_shape[3];
    int64_t sH = strides.size() > 0 ? strides[0] : 1, sW = strides.size() > 1 ? strides[1] : 1;
    int64_t dH = dilations.size() > 0 ? dilations[0] : 1, dW = dilations.size() > 1 ? dilations[1] : 1;
    int64_t pH = pads_begin.size() > 0 ? pads_begin[0] : 0, pW = pads_begin.size() > 1 ? pads_begin[1] : 0;
    int64_t oH = (iH - 1) * sH - 2 * pH + dH * (kH - 1) + 1;
    int64_t oW = (iW - 1) * sW - 2 * pW + dW * (kW - 1) + 1;
    int64_t ic_per_group = IC / groups, oc_per_group = OC / groups;
    std::vector<float> output(N * OC * oH * oW, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < groups; ++g)
            for (int64_t ic = 0; ic < ic_per_group; ++ic)
                for (int64_t oc = 0; oc < oc_per_group; ++oc)
                    for (int64_t ih = 0; ih < iH; ++ih)
                        for (int64_t iw = 0; iw < iW; ++iw)
                            for (int64_t kh = 0; kh < kH; ++kh)
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t oh = ih * sH + kh * dH - pH;
                                    int64_t ow = iw * sW + kw * dW - pW;
                                    if (oh < 0 || oh >= oH || ow < 0 || ow >= oW) continue;
                                    float in_val = input[((n * IC + g * ic_per_group + ic) * iH + ih) * iW + iw];
                                    float w_val = weights[(((g * ic_per_group + ic) * oc_per_group + oc) * kH + kh) * kW + kw];
                                    output[((n * OC + g * oc_per_group + oc) * oH + oh) * oW + ow] += in_val * w_val;
                                }
    return output;
}

// ============================================================================
// RegionYolo reference
// ============================================================================
inline std::vector<float> region_yolo(const std::vector<float>& input,
                                      const std::vector<int64_t>& shape,
                                      int coords, int classes, int num,
                                      bool do_softmax) {
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    std::vector<float> output(input);
    int entry_size = coords + 1 + classes;
    for (int64_t n = 0; n < N; ++n)
        for (int a = 0; a < num; ++a)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w) {
                    int base = a * entry_size;
                    // sigmoid x, y
                    for (int i = 0; i < 2; ++i) {
                        size_t idx = ((n * C + base + i) * H + h) * W + w;
                        output[idx] = 1.0f / (1.0f + std::exp(-output[idx]));
                    }
                    // sigmoid confidence
                    {
                        size_t idx = ((n * C + base + coords) * H + h) * W + w;
                        output[idx] = 1.0f / (1.0f + std::exp(-output[idx]));
                    }
                    // softmax or sigmoid classes
                    if (do_softmax) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int c = 0; c < classes; ++c) {
                            size_t idx = ((n * C + base + coords + 1 + c) * H + h) * W + w;
                            max_val = std::max(max_val, output[idx]);
                        }
                        float sum = 0;
                        for (int c = 0; c < classes; ++c) {
                            size_t idx = ((n * C + base + coords + 1 + c) * H + h) * W + w;
                            output[idx] = std::exp(output[idx] - max_val);
                            sum += output[idx];
                        }
                        for (int c = 0; c < classes; ++c) {
                            size_t idx = ((n * C + base + coords + 1 + c) * H + h) * W + w;
                            output[idx] /= sum;
                        }
                    } else {
                        for (int c = 0; c < classes; ++c) {
                            size_t idx = ((n * C + base + coords + 1 + c) * H + h) * W + w;
                            output[idx] = 1.0f / (1.0f + std::exp(-output[idx]));
                        }
                    }
                }
    return output;
}

// ============================================================================
// LSTM Cell reference
// ============================================================================
inline std::vector<float> lstm_cell(const std::vector<float>& x,
                                    const std::vector<float>& h_prev,
                                    const std::vector<float>& c_prev,
                                    const std::vector<float>& W,
                                    const std::vector<float>& R,
                                    const std::vector<float>& B,
                                    int64_t batch, int64_t input_size, int64_t hidden_size) {
    // Gates order: i, o, f, z (same as iofc/iofz)
    // H_new and C_new output concatenated: [batch * hidden_size (h), batch * hidden_size (c)]
    std::vector<float> gates(batch * 4 * hidden_size, 0.0f);
    // gates = x*W^T + h*R^T + B
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t g = 0; g < 4 * hidden_size; ++g) {
            float val = B.size() > static_cast<size_t>(g) ? B[g] : 0.0f;
            for (int64_t i = 0; i < input_size; ++i)
                val += x[b * input_size + i] * W[g * input_size + i];
            for (int64_t i = 0; i < hidden_size; ++i)
                val += h_prev[b * hidden_size + i] * R[g * hidden_size + i];
            gates[b * 4 * hidden_size + g] = val;
        }
    std::vector<float> output(batch * hidden_size * 2);
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t h = 0; h < hidden_size; ++h) {
            float i_val = 1.0f / (1.0f + std::exp(-gates[b * 4 * hidden_size + h]));               // sigmoid
            float o_val = 1.0f / (1.0f + std::exp(-gates[b * 4 * hidden_size + hidden_size + h])); // sigmoid
            float f_val = 1.0f / (1.0f + std::exp(-gates[b * 4 * hidden_size + 2 * hidden_size + h])); // sigmoid
            float z_val = std::tanh(gates[b * 4 * hidden_size + 3 * hidden_size + h]);             // tanh
            float c_new = f_val * c_prev[b * hidden_size + h] + i_val * z_val;
            float h_new = o_val * std::tanh(c_new);
            output[b * hidden_size + h] = h_new;
            output[batch * hidden_size + b * hidden_size + h] = c_new;
        }
    return output;
}

// ============================================================================
// ConvertColor reference (NV12 to BGR/RGB)
// ============================================================================
inline std::vector<float> convert_color_nv12_to_bgr(const std::vector<float>& input,
                                                     const std::vector<int64_t>& shape) {
    // input: [N, H*3/2, W, 1], output: [N, H, W, 3]
    int64_t N = shape[0], H_full = shape[1], W = shape[2];
    int64_t H = H_full * 2 / 3; // Y plane height
    std::vector<float> output(N * H * W * 3);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w) {
                float Y = input[(n * H_full + h) * W + w];
                float U = input[(n * H_full + H + h / 2) * W + (w / 2) * 2];
                float V = input[(n * H_full + H + h / 2) * W + (w / 2) * 2 + 1];
                float R = Y + 1.402f * (V - 128.0f);
                float G = Y - 0.344136f * (U - 128.0f) - 0.714136f * (V - 128.0f);
                float B = Y + 1.772f * (U - 128.0f);
                size_t out_idx = ((n * H + h) * W + w) * 3;
                output[out_idx + 0] = std::max(0.0f, std::min(255.0f, B));
                output[out_idx + 1] = std::max(0.0f, std::min(255.0f, G));
                output[out_idx + 2] = std::max(0.0f, std::min(255.0f, R));
            }
    return output;
}

// ============================================================================
// NonZero reference
// ============================================================================
inline std::vector<float> non_zero(const std::vector<float>& input,
                                   const std::vector<int64_t>& shape) {
    size_t rank = shape.size();
    size_t total = total_elements(shape);
    std::vector<size_t> strides(rank, 1);
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * shape[d + 1];
    // Find nonzero coordinates
    std::vector<std::vector<int64_t>> coords;
    for (size_t i = 0; i < total; ++i) {
        if (input[i] != 0.0f) {
            std::vector<int64_t> c(rank);
            size_t tmp = i;
            for (size_t d = 0; d < rank; ++d) { c[d] = tmp / strides[d]; tmp %= strides[d]; }
            coords.push_back(c);
        }
    }
    // Output: [rank, num_nonzero]
    size_t num_nz = coords.size();
    std::vector<float> output(rank * num_nz);
    for (size_t d = 0; d < rank; ++d)
        for (size_t i = 0; i < num_nz; ++i)
            output[d * num_nz + i] = static_cast<float>(coords[i][d]);
    return output;
}

// ============================================================================
// DFT reference (simplified: 1D DFT on last axis)
// ============================================================================
inline std::vector<float> dft(const std::vector<float>& input,
                              const std::vector<int64_t>& shape,
                              const std::vector<int64_t>& axes,
                              bool inverse, int64_t signal_size = -1) {
    // Simplified DFT on flattened complex pairs
    // For real DFT: input is real, output has complex pairs
    // For full complex: last dim has 2 (real, imag)
    size_t total = input.size();
    std::vector<float> output(total, 0.0f);
    // Apply 1D DFT along each specified axis (cascading)
    size_t rank = shape.size();
    bool is_complex = (shape.back() == 2);
    std::vector<float> current = input;  // working buffer, cascades across axes
    for (auto ax : axes) {
        if (ax < 0) ax += static_cast<int64_t>(rank);
        int64_t N_dft = shape[ax];
        if (signal_size > 0) N_dft = signal_size;
        size_t before = 1, after = 1;
        for (int64_t d = 0; d < ax; ++d) before *= shape[d];
        for (size_t d = ax + 1; d < rank; ++d) after *= shape[d];
        std::vector<float> tmp(total);
        size_t complex_factor = is_complex ? 2 : 1;
        after /= complex_factor;
        for (size_t b = 0; b < before; ++b)
            for (size_t a = 0; a < after; ++a)
                for (int64_t k = 0; k < N_dft; ++k) {
                    float re = 0, im = 0;
                    float sign = inverse ? 1.0f : -1.0f;
                    for (int64_t n = 0; n < N_dft; ++n) {
                        float angle = sign * 2.0f * 3.14159265358979f * k * n / N_dft;
                        float cos_a = std::cos(angle), sin_a = std::sin(angle);
                        size_t src = (b * N_dft + n) * after * complex_factor + a * complex_factor;
                        float src_re = (src < total) ? current[src] : 0.0f;
                        float src_im = is_complex && (src + 1 < total) ? current[src + 1] : 0.0f;
                        re += src_re * cos_a - src_im * sin_a;
                        im += src_re * sin_a + src_im * cos_a;
                    }
                    if (inverse) { re /= N_dft; im /= N_dft; }
                    size_t dst = (b * N_dft + k) * after * complex_factor + a * complex_factor;
                    if (dst < total) tmp[dst] = re;
                    if (is_complex && dst + 1 < total) tmp[dst + 1] = im;
                }
        current = tmp;  // cascade: next axis reads from this axis's result
    }
    return current;
}

// ============================================================================
// Range check for random/non-deterministic ops
// ============================================================================
inline std::vector<float> make_range_check_ref(const std::vector<float>& gpu_out,
                                               float min_val, float max_val) {
    // Returns a vector same size as gpu_out, with each element clamped to [min, max]
    // If all values are in range, compare_f32 will pass with atol=0
    std::vector<float> ref(gpu_out.size());
    for (size_t i = 0; i < gpu_out.size(); ++i)
        ref[i] = std::max(min_val, std::min(max_val, gpu_out[i]));
    return ref;
}

// ============================================================================
// CTC Greedy Decoder reference
// ============================================================================
inline std::vector<float> ctc_greedy_decoder(const std::vector<float>& logits,
                                             const std::vector<float>& seq_lens,
                                             const std::vector<int64_t>& shape) {
    int64_t T = shape[0], N = shape[1], C = shape[2];
    std::vector<float> output(T * N, -1.0f);
    for (int64_t n = 0; n < N; ++n) {
        int64_t sl = static_cast<int64_t>(seq_lens[n]);
        int64_t prev = -1, out_idx = 0;
        for (int64_t t = 0; t < std::min(T, sl); ++t) {
            // Find argmax
            int64_t best = 0;
            float best_val = logits[(t * N + n) * C];
            for (int64_t c = 1; c < C; ++c) {
                float v = logits[(t * N + n) * C + c];
                if (v > best_val) { best_val = v; best = c; }
            }
            int64_t blank = C - 1; // blank is last class
            if (best != blank && best != prev) {
                output[out_idx * N + n] = static_cast<float>(best);
                out_idx++;
            }
            prev = best;
        }
    }
    return output;
}

// ============================================================================
// PriorBox reference (simplified)
// ============================================================================
inline std::vector<float> prior_box(const std::vector<int64_t>& feat_shape,
                                    const std::vector<int64_t>& img_shape,
                                    const std::vector<float>& min_sizes,
                                    const std::vector<float>& max_sizes,
                                    const std::vector<float>& aspect_ratios,
                                    const std::vector<float>& variance,
                                    float step, float offset, bool flip, bool clip) {
    int64_t fH = feat_shape[0], fW = feat_shape[1];
    int64_t imgH = img_shape[0], imgW = img_shape[1];
    float step_h = (step == 0) ? static_cast<float>(imgH) / fH : step;
    float step_w = (step == 0) ? static_cast<float>(imgW) / fW : step;
    // Expand aspect ratios
    std::vector<float> ars = {1.0f};
    for (auto ar : aspect_ratios) {
        if (std::abs(ar - 1.0f) > 1e-6f) { ars.push_back(ar); if (flip) ars.push_back(1.0f / ar); }
    }
    // Count boxes per cell
    int num_priors = 0;
    for (size_t i = 0; i < min_sizes.size(); ++i) {
        num_priors += static_cast<int>(ars.size());
        if (i < max_sizes.size()) num_priors++;
    }
    std::vector<float> output;
    for (int64_t h = 0; h < fH; ++h)
        for (int64_t w = 0; w < fW; ++w) {
            float cx = (w + offset) * step_w;
            float cy = (h + offset) * step_h;
            for (size_t i = 0; i < min_sizes.size(); ++i) {
                float ms = min_sizes[i];
                for (auto ar : ars) {
                    float box_w = ms * std::sqrt(ar);
                    float box_h = ms / std::sqrt(ar);
                    float x1 = (cx - box_w / 2.0f) / imgW;
                    float y1 = (cy - box_h / 2.0f) / imgH;
                    float x2 = (cx + box_w / 2.0f) / imgW;
                    float y2 = (cy + box_h / 2.0f) / imgH;
                    if (clip) { x1 = std::max(0.0f, std::min(1.0f, x1)); y1 = std::max(0.0f, std::min(1.0f, y1));
                                x2 = std::max(0.0f, std::min(1.0f, x2)); y2 = std::max(0.0f, std::min(1.0f, y2)); }
                    output.push_back(x1); output.push_back(y1); output.push_back(x2); output.push_back(y2);
                }
                if (i < max_sizes.size()) {
                    float ms2 = std::sqrt(ms * max_sizes[i]);
                    float x1 = (cx - ms2 / 2.0f) / imgW;
                    float y1 = (cy - ms2 / 2.0f) / imgH;
                    float x2 = (cx + ms2 / 2.0f) / imgW;
                    float y2 = (cy + ms2 / 2.0f) / imgH;
                    if (clip) { x1 = std::max(0.0f, std::min(1.0f, x1)); y1 = std::max(0.0f, std::min(1.0f, y1));
                                x2 = std::max(0.0f, std::min(1.0f, x2)); y2 = std::max(0.0f, std::min(1.0f, y2)); }
                    output.push_back(x1); output.push_back(y1); output.push_back(x2); output.push_back(y2);
                }
            }
        }
    // Variance layer
    size_t num_boxes = output.size() / 4;
    for (size_t i = 0; i < num_boxes; ++i)
        for (auto v : variance)
            output.push_back(v);
    return output;
}

// ============================================================================
// SparseFillEmptyRows reference
// ============================================================================
inline std::vector<float> sparse_fill_empty_rows(const std::vector<float>& indices_f32,
                                                 const std::vector<float>& values,
                                                 int64_t num_rows, int64_t N,
                                                 float default_value) {
    // Check which rows have entries
    std::vector<bool> has_entry(num_rows, false);
    for (int64_t i = 0; i < N; ++i) {
        int64_t row = static_cast<int64_t>(indices_f32[i * 2]);
        if (row >= 0 && row < num_rows) has_entry[row] = true;
    }
    // Build output: existing entries + filled empty rows
    // Output indices: [M, 2], output values: [M]
    std::vector<float> out_values;
    for (int64_t i = 0; i < N; ++i) out_values.push_back(values[i]);
    for (int64_t r = 0; r < num_rows; ++r)
        if (!has_entry[r]) out_values.push_back(default_value);
    return out_values;
}

}  // namespace ref

}  // namespace bench_kernel
