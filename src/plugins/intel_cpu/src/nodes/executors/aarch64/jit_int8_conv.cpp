// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_int8_conv.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include <cpu/aarch64/brgemm/brgemm_types.hpp>

#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "openvino/core/parallel.hpp"
#include "post_ops.hpp"
#include "utils/precision_support.h"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::aarch64 {

namespace {
using dnnl::impl::cpu::aarch64::brgemm_batch_element_t;

const MappingNotation kConvMappingNotation = {{ARG_SRC, 0}, {ARG_WEI, 1}, {ARG_BIAS, 2}, {ARG_DST, 3}};

bool is_zero(const std::vector<size_t>& values) {
    return std::all_of(values.begin(), values.end(), [](size_t v) {
        return v == 0;
    });
}

bool is_non_negative(const std::vector<ptrdiff_t>& values) {
    return std::all_of(values.begin(), values.end(), [](ptrdiff_t v) {
        return v >= 0;
    });
}

const FakeQuantizePostOp* getFqPostOp(const PostOps& ops) {
    if (ops.size() != 1) {
        return nullptr;
    }
    return std::any_cast<FakeQuantizePostOp>(ops.data());
}

bool isSupportedFq(const FakeQuantizePostOp& fq, size_t oc) {
    const auto& in_scale = fq.inputScale();
    const auto& in_shift = fq.inputShift();
    const auto& out_scale = fq.outputScale();
    const auto& out_shift = fq.outputShift();
    const auto& crop_low = fq.cropLow();
    const auto& crop_high = fq.cropHigh();
    const auto in_scale_ok = in_scale.empty() || in_scale.size() == 1 || in_scale.size() == oc;
    const auto in_shift_ok = in_shift.empty() || in_shift.size() == 1 || in_shift.size() == oc;
    const auto crop_low_ok = crop_low.empty() || crop_low.size() == 1 || crop_low.size() == oc;
    const auto crop_high_ok = crop_high.empty() || crop_high.size() == 1 || crop_high.size() == oc;
    if (!in_scale_ok || !in_shift_ok) {
        return false;
    }
    if (!crop_low_ok || !crop_high_ok) {
        return false;
    }
    if (!out_scale.empty() && !(out_scale.size() == 1 || out_scale.size() == oc)) {
        return false;
    }
    if (!out_shift.empty() && !(out_shift.size() == 1 || out_shift.size() == oc)) {
        return false;
    }
    return true;
}

float getBroadcasted(const std::vector<float>& values, size_t idx, float fallback) {
    if (values.empty()) {
        return fallback;
    }
    if (values.size() == 1) {
        return values[0];
    }
    return values[idx];
}

float getBiasValue(const std::vector<float>& values, size_t idx) {
    if (values.empty()) {
        return 0.0f;
    }
    if (values.size() == 1) {
        return values[0];
    }
    return values[idx];
}

[[maybe_unused]] std::string postOpsToString(const PostOps& ops) {
    if (ops.empty()) {
        return "none";
    }
    std::string out;
    for (size_t i = 0; i < ops.size(); ++i) {
        if (i) {
            out += "+";
        }
        if (std::any_cast<SumPostOp>(&ops[i])) {
            out += "Sum";
        } else if (std::any_cast<FakeQuantizePostOp>(&ops[i])) {
            out += "FakeQuantize";
        } else if (std::any_cast<ActivationPostOp>(&ops[i])) {
            out += "Activation";
        } else if (std::any_cast<ScaleShiftPostOp>(&ops[i])) {
            out += "ScaleShift";
        } else if (std::any_cast<DepthwiseConvolutionPostOp>(&ops[i])) {
            out += "DepthwiseConv";
        } else {
            out += "Unknown";
        }
    }
    return out;
}

bool jit_int8_conv_debug_enabled() {
    static const bool enabled = std::getenv("OV_CPU_JIT_INT8_CONV_DEBUG") != nullptr;
    return enabled;
}

template <typename T>
void append_to_stream(std::ostringstream& oss, const T& value) {
    oss << value;
}

template <typename T>
void append_to_stream(std::ostringstream& oss, const std::vector<T>& values) {
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            oss << ",";
        }
        oss << values[i];
    }
    oss << "]";
}

template <typename... Ts>
void jit_int8_conv_debug(Ts&&... args) {
    if (!jit_int8_conv_debug_enabled()) {
        return;
    }
    std::ostringstream oss;
    (append_to_stream(oss, std::forward<Ts>(args)), ...);
    std::cerr << "[jit_int8_conv] " << oss.str() << '\n';
}

const char* fallback_family_name_for_debug(bool has_i8mm, bool has_dotprod) {
    if (has_i8mm) {
        return "custom_neon_i8mm";
    }
    if (has_dotprod) {
        return "custom_neon_dotprod";
    }
    return "reference_cpp";
}

inline void add_1x1_block_2x32(int32_t* dst_ptr, size_t row_stride, const int32_t* add_ptr) {
    int32_t* dst_row0 = dst_ptr;
    int32_t* dst_row1 = dst_row0 + row_stride;
    for (size_t c = 0; c < 32; ++c) {
        const int32_t v = add_ptr[c];
        dst_row0[c] += v;
        dst_row1[c] += v;
    }
}

inline void add_1x1_block_2x16(int32_t* dst_ptr, size_t row_stride, const int32_t* add_ptr) {
    int32_t* dst_row0 = dst_ptr;
    int32_t* dst_row1 = dst_row0 + row_stride;
    for (size_t c = 0; c < 16; ++c) {
        const int32_t v = add_ptr[c];
        dst_row0[c] += v;
        dst_row1[c] += v;
    }
}

template <typename DstT>
void requantize_fq(const PlainTensor& src_i32,
                   PlainTensor& dst,
                   const std::vector<float>& dq_scales,
                   const FakeQuantizePostOp& fq,
                   const std::vector<float>* bias) {
    const auto& dims = dst.shape();
    if (dims.size() != 4) {
        return;
    }
    const size_t N = dims[0];
    const size_t OC = dims[1];
    const size_t H = dims[2];
    const size_t W = dims[3];

    const auto& in_scale = fq.inputScale();
    const auto& in_shift = fq.inputShift();
    const auto& out_scale = fq.outputScale();
    const auto& out_shift = fq.outputShift();
    const auto& crop_low = fq.cropLow();
    const auto& crop_high = fq.cropHigh();
    const float q_levels = static_cast<float>(fq.levels());

    float extra_shift = 0.0f;
    if (!out_shift.empty()) {
        extra_shift = out_shift[0];
    }

    const int32_t clamp_min = std::numeric_limits<DstT>::min();
    const int32_t clamp_max = std::numeric_limits<DstT>::max();

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                const int32_t* src_ptr = src_i32.ptr<int32_t>(n, 0, h, w);
                DstT* dst_ptr = dst.ptr<DstT>(n, 0, h, w);
                for (size_t oc = 0; oc < OC; ++oc) {
                    const float w_scale = getBroadcasted(dq_scales, oc, 1.0f);
                    const float bias_val = bias ? getBiasValue(*bias, oc) : 0.0f;
                    const float x = static_cast<float>(src_ptr[oc]) * w_scale + bias_val;
                    const float clo = getBroadcasted(crop_low, oc, -std::numeric_limits<float>::infinity());
                    const float chi = getBroadcasted(crop_high, oc, std::numeric_limits<float>::infinity());
                    const float x_clamped = std::min(std::max(x, clo), chi);
                    const float q_scale = getBroadcasted(in_scale, oc, 1.0f);
                    const float q_shift = getBroadcasted(in_shift, oc, 0.0f);
                    float q = std::nearbyint(x_clamped * q_scale + q_shift);
                    if (q_levels > 1.0f) {
                        q = std::min(std::max(q, 0.0f), q_levels - 1.0f);
                    }
                    const float o_scale = getBroadcasted(out_scale, oc, 1.0f);
                    const float o_shift = getBroadcasted(out_shift, oc, 0.0f) + extra_shift;
                    const float y = q * o_scale + o_shift;
                    const int32_t q_out = static_cast<int32_t>(std::nearbyint(y));
                    const int32_t clamped = std::min(std::max(q_out, clamp_min), clamp_max);
                    dst_ptr[oc] = static_cast<DstT>(clamped);
                }
            }
        }
    }
}

template <typename DstT>
void requantize_simple(const PlainTensor& src_i32,
                       PlainTensor& dst,
                       const std::vector<float>& dq_scales,
                       const std::vector<float>* bias) {
    const auto& dims = dst.shape();
    if (dims.size() != 4) {
        return;
    }
    const size_t N = dims[0];
    const size_t OC = dims[1];
    const size_t H = dims[2];
    const size_t W = dims[3];
    const int32_t clamp_min = std::numeric_limits<DstT>::min();
    const int32_t clamp_max = std::numeric_limits<DstT>::max();
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                const int32_t* src_ptr = src_i32.ptr<int32_t>(n, 0, h, w);
                DstT* dst_ptr = dst.ptr<DstT>(n, 0, h, w);
                for (size_t oc = 0; oc < OC; ++oc) {
                    const float scale = getBroadcasted(dq_scales, oc, 1.0f);
                    const float bias_val = bias ? getBiasValue(*bias, oc) : 0.0f;
                    const float val = static_cast<float>(src_ptr[oc]) * scale + bias_val;
                    const int32_t q = static_cast<int32_t>(std::nearbyint(val));
                    const int32_t clamped = std::min(std::max(q, clamp_min), clamp_max);
                    dst_ptr[oc] = static_cast<DstT>(clamped);
                }
            }
        }
    }
}

size_t packed_block_stride(size_t K, size_t oc_block) {
    const size_t k_blocks = K / 16;
    const size_t k_tail = K % 16;
    return k_blocks * oc_block * 16 + k_tail * oc_block;
}

size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

size_t packed_block_stride_dot_interleaved(size_t K, size_t oc_block) {
    const size_t Kp = round_up(K, 4);
    return Kp * oc_block;
}

size_t packed_block_stride_mmla(size_t K, size_t oc_block) {
    const size_t Kp = round_up(K, 8);
    return (Kp / 8) * oc_block * 8;
}

void pack_dot_block(const int8_t* src, size_t K, size_t oc_block, int8_t* dst) {
    const size_t k_blocks = K / 16;
    const size_t k_tail = K % 16;
    size_t offset = 0;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_off = kb * 16;
        for (size_t oc = 0; oc < oc_block; ++oc) {
            const int8_t* s = src + oc * K + k_off;
            std::copy_n(s, 16, dst + offset);
            offset += 16;
        }
    }
    const size_t k_base = k_blocks * 16;
    for (size_t kt = 0; kt < k_tail; ++kt) {
        for (size_t oc = 0; oc < oc_block; ++oc) {
            dst[offset++] = src[oc * K + k_base + kt];
        }
    }
}

void pack_dot_block_interleaved4(const int8_t* src, size_t K, size_t oc_block, int8_t* dst) {
    const size_t k_blocks = round_up(K, 4) / 4;
    size_t offset = 0;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_off = kb * 4;
        for (size_t oc = 0; oc < oc_block; oc += 4) {
            for (size_t lane = 0; lane < 4; ++lane) {
                const size_t oc_idx = oc + lane;
                for (size_t t = 0; t < 4; ++t) {
                    const size_t k = k_off + t;
                    const size_t dst_idx = offset + lane * 4 + t;
                    if (oc_idx < oc_block && k < K) {
                        dst[dst_idx] = src[oc_idx * K + k];
                    } else {
                        dst[dst_idx] = 0;
                    }
                }
            }
            offset += 16;
        }
    }
}

void pack_mmla_block(const int8_t* src, size_t K, size_t Kp, size_t oc_block, int8_t* dst) {
    const size_t k_blocks = Kp / 8;
    size_t offset = 0;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_off = kb * 8;
        const size_t valid = k_off < K ? std::min<size_t>(8, K - k_off) : 0;
        for (size_t oc = 0; oc + 1 < oc_block; oc += 2) {
            const int8_t* s0 = src + oc * K + k_off;
            const int8_t* s1 = src + (oc + 1) * K + k_off;
            if (valid == 8) {
                std::memcpy(dst + offset, s0, 8);
                std::memcpy(dst + offset + 8, s1, 8);
            } else {
                std::memset(dst + offset, 0, 16);
                if (valid > 0) {
                    std::memcpy(dst + offset, s0, valid);
                    std::memcpy(dst + offset + 8, s1, valid);
                }
            }
            offset += 16;
        }
    }
}

inline void pack_pair_8bytes(const uint8_t* src0, const uint8_t* src1, uint8_t* dst, size_t ic_blocks) {
    auto* dst64 = reinterpret_cast<uint64_t*>(dst);
    auto* src0_64 = reinterpret_cast<const uint64_t*>(src0);
    auto* src1_64 = reinterpret_cast<const uint64_t*>(src1);
    size_t icb = 0;
    for (; icb + 1 < ic_blocks; icb += 2) {
        const uint64_t s0_0 = src0_64[0];
        const uint64_t s1_0 = src1_64[0];
        const uint64_t s0_1 = src0_64[1];
        const uint64_t s1_1 = src1_64[1];
        dst64[0] = s0_0;
        dst64[1] = s1_0;
        dst64[2] = s0_1;
        dst64[3] = s1_1;
        src0_64 += 2;
        src1_64 += 2;
        dst64 += 4;
    }
    for (; icb < ic_blocks; ++icb) {
        dst64[0] = *src0_64++;
        dst64[1] = *src1_64++;
        dst64 += 2;
    }
}

inline void pack_pairs4x8(const uint8_t* src0,
                          const uint8_t* src1,
                          const uint8_t* src2,
                          const uint8_t* src3,
                          uint8_t* dst0,
                          uint8_t* dst1,
                          size_t ic_blocks) {
    // Reuse the 2x8 packer: it is unrolled and uses post-increment pointers.
    pack_pair_8bytes(src0, src1, dst0, ic_blocks);
    pack_pair_8bytes(src2, src3, dst1, ic_blocks);
}

inline void pack_pairs8x8(const uint8_t* src0,
                          const uint8_t* src1,
                          const uint8_t* src2,
                          const uint8_t* src3,
                          const uint8_t* src4,
                          const uint8_t* src5,
                          const uint8_t* src6,
                          const uint8_t* src7,
                          uint8_t* dst0,
                          uint8_t* dst1,
                          uint8_t* dst2,
                          uint8_t* dst3,
                          size_t ic_blocks) {
    // Reuse the 2x8 packer: it is unrolled and uses post-increment pointers.
    pack_pair_8bytes(src0, src1, dst0, ic_blocks);
    pack_pair_8bytes(src2, src3, dst1, ic_blocks);
    pack_pair_8bytes(src4, src5, dst2, ic_blocks);
    pack_pair_8bytes(src6, src7, dst3, ic_blocks);
}

inline void pack_lhs_row_dot4x16_interleaved4(const uint8_t* src, size_t K, uint8_t* dst) {
    const size_t k_blocks = K >> 2;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_base = kb * 4;
        const uint8_t v0 = src[k_base];
        const uint8_t v1 = src[k_base + 1];
        const uint8_t v2 = src[k_base + 2];
        const uint8_t v3 = src[k_base + 3];
        const uint32_t r0 = static_cast<uint32_t>(v0) * 0x01010101u;
        const uint32_t r1 = static_cast<uint32_t>(v1) * 0x01010101u;
        const uint32_t r2 = static_cast<uint32_t>(v2) * 0x01010101u;
        const uint32_t r3 = static_cast<uint32_t>(v3) * 0x01010101u;
        std::memcpy(dst, &r0, sizeof(r0));
        std::memcpy(dst + 4, &r1, sizeof(r1));
        std::memcpy(dst + 8, &r2, sizeof(r2));
        std::memcpy(dst + 12, &r3, sizeof(r3));
        dst += 16;
    }
}

inline void pack_lhs_4x16_dot_interleaved4(const uint8_t* src0,
                                           const uint8_t* src1,
                                           const uint8_t* src2,
                                           const uint8_t* src3,
                                           size_t K,
                                           uint8_t* dst,
                                           size_t dst_stride) {
    pack_lhs_row_dot4x16_interleaved4(src0, K, dst);
    pack_lhs_row_dot4x16_interleaved4(src1, K, dst + dst_stride);
    pack_lhs_row_dot4x16_interleaved4(src2, K, dst + 2 * dst_stride);
    pack_lhs_row_dot4x16_interleaved4(src3, K, dst + 3 * dst_stride);
}

inline void prefetch_l1(const void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);
#else
    (void)ptr;
#endif
}
}  // namespace

BrgemmInt8ConvExecutor::BrgemmInt8ConvExecutor(const ConvAttrs& attrs,
                                               const MemoryArgs&,
                                               const ExecutorContext::CPtr&)
    : attrs_(attrs),
      kernel_u8_(std::make_shared<jit_int8_dot_kernel>(false)),
      kernel_s8_(std::make_shared<jit_int8_dot_kernel>(true)),
      kernel_block4_u8_(std::make_shared<jit_int8_brgemm_kernel_1x4>(false)),
      kernel_block4_s8_(std::make_shared<jit_int8_brgemm_kernel_1x4>(true)),
      kernel_block4_dot_s8_(std::make_shared<jit_int8_brgemm_kernel_1x4_dot>()),
      kernel_block4_udot_(std::make_shared<jit_int8_brgemm_kernel_1x4_udot>()),
      kernel_block8_dot_s8_(std::make_shared<jit_int8_brgemm_kernel_1x8_dot>()),
      kernel_block8_udot_(std::make_shared<jit_int8_brgemm_kernel_1x8_udot>()),
      kernel_block8_dot_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_1x8_dot_packed>()),
      kernel_block8_udot_packed_(std::make_shared<jit_int8_brgemm_kernel_1x8_udot_packed>()),
      kernel_block2x8_dot_packed_s8_strided_(std::make_shared<jit_int8_brgemm_kernel_2x8_dot_packed_strided>()),
      kernel_block2x8_dot_packed_s8_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_2x8_dot_packed_strided_interleaved4>()),
      kernel_block2x16_dot_packed_s8_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_2x16_dot_packed_strided_interleaved4>()),
      kernel_block2x32_dot_packed_s8_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_2x32_dot_packed_strided_interleaved4>()),
      kernel_block4x16_dot_packed_s8_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_4x16_dot_packed_strided_interleaved4>()),
      kernel_block4x16_dot_packed_lhs_s8_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_4x16_dot_packed_lhs_strided_interleaved4>()),
      kernel_block2x8_udot_packed_strided_(std::make_shared<jit_int8_brgemm_kernel_2x8_udot_packed_strided>()),
      kernel_block2x8_udot_packed_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_2x8_udot_packed_strided_interleaved4>()),
      kernel_block2x16_udot_packed_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_2x16_udot_packed_strided_interleaved4>()),
      kernel_block2x32_udot_packed_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_2x32_udot_packed_strided_interleaved4>()),
      kernel_block4x16_udot_packed_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_4x16_udot_packed_strided_interleaved4>()),
      kernel_block4x16_udot_packed_lhs_strided_interleaved_(
          std::make_shared<jit_int8_brgemm_kernel_4x16_udot_packed_lhs_strided_interleaved4>()),
      kernel_block4x4_dot_s8_(std::make_shared<jit_int8_brgemm_kernel_4x4_dot>()),
      kernel_block4x4_mmla_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_4x4_smmla_packed>()),
      kernel_block4x8_mmla_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_4x8_smmla_packed>()),
      kernel_block4x16_mmla_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_4x16_smmla_packed>()),
      kernel_block4x4_mmla_packed_s8_interleaved_(std::make_shared<jit_int8_brgemm_kernel_4x4_smmla_packed>(true)),
      kernel_block4x8_mmla_packed_s8_interleaved_(std::make_shared<jit_int8_brgemm_kernel_4x8_smmla_packed>(true)),
      kernel_block8x8_mmla_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_8x8_smmla_packed>()),
      kernel_block8x12_mmla_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_8x12_smmla_packed>()),
      kernel_block4x16_mmla_packed_s8_interleaved_(std::make_shared<jit_int8_brgemm_kernel_4x16_smmla_packed>(true)),
      kernel_block4x4_udot_(std::make_shared<jit_int8_brgemm_kernel_4x4_udot>()),
      kernel_block4x4_dot_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_4x4_dot_packed>()),
      kernel_block4x4_mmla_packed_u8_(std::make_shared<jit_int8_brgemm_kernel_4x4_usmmla_packed>()),
      kernel_block4x8_mmla_packed_u8_(std::make_shared<jit_int8_brgemm_kernel_4x8_usmmla_packed>()),
      kernel_block4x16_mmla_packed_u8_(std::make_shared<jit_int8_brgemm_kernel_4x16_usmmla_packed>()),
      kernel_block4x4_mmla_packed_u8_interleaved_(std::make_shared<jit_int8_brgemm_kernel_4x4_usmmla_packed>(true)),
      kernel_block4x8_mmla_packed_u8_interleaved_(std::make_shared<jit_int8_brgemm_kernel_4x8_usmmla_packed>(true)),
      kernel_block8x8_mmla_packed_u8_(std::make_shared<jit_int8_brgemm_kernel_8x8_usmmla_packed>()),
      kernel_block8x12_mmla_packed_u8_(std::make_shared<jit_int8_brgemm_kernel_8x12_usmmla_packed>()),
      kernel_block4x16_mmla_packed_u8_interleaved_(std::make_shared<jit_int8_brgemm_kernel_4x16_usmmla_packed>(true)),
      kernel_block4x4_udot_packed_(std::make_shared<jit_int8_brgemm_kernel_4x4_udot_packed>()),
      runtime_isa_(ov::intel_cpu::getAarch64Int8Isa()),
      has_dotprod_(ov::intel_cpu::hasIntDotProductSupport()),
      has_i8mm_(ov::intel_cpu::hasInt8MMSupport()) {
    jit_int8_conv_debug("runtime_isa=", ov::intel_cpu::aarch64Int8IsaName(runtime_isa_),
                        " dotprod=", has_dotprod_,
                        " i8mm=", has_i8mm_,
                        " sve=", ov::intel_cpu::hasSVESupport());
    kernel_u8_->create_ker();
    kernel_s8_->create_ker();
    kernel_block4_u8_->create_ker();
    kernel_block4_s8_->create_ker();
    kernel_block4_dot_s8_->create_ker();
    kernel_block4_udot_->create_ker();
    kernel_block8_dot_s8_->create_ker();
    kernel_block8_udot_->create_ker();
    kernel_block8_dot_packed_s8_->create_ker();
    kernel_block8_udot_packed_->create_ker();
    kernel_block2x8_dot_packed_s8_strided_->create_ker();
    kernel_block2x8_dot_packed_s8_strided_interleaved_->create_ker();
    kernel_block2x16_dot_packed_s8_strided_interleaved_->create_ker();
    kernel_block2x32_dot_packed_s8_strided_interleaved_->create_ker();
    kernel_block4x16_dot_packed_s8_strided_interleaved_->create_ker();
    kernel_block4x16_dot_packed_lhs_s8_strided_interleaved_->create_ker();
    kernel_block2x8_udot_packed_strided_->create_ker();
    kernel_block2x8_udot_packed_strided_interleaved_->create_ker();
    kernel_block2x16_udot_packed_strided_interleaved_->create_ker();
    kernel_block2x32_udot_packed_strided_interleaved_->create_ker();
    kernel_block4x16_udot_packed_strided_interleaved_->create_ker();
    kernel_block4x16_udot_packed_lhs_strided_interleaved_->create_ker();
    kernel_block4x4_dot_s8_->create_ker();
    kernel_block4x4_mmla_packed_s8_->create_ker();
    kernel_block4x8_mmla_packed_s8_->create_ker();
    kernel_block4x16_mmla_packed_s8_->create_ker();
    kernel_block4x4_mmla_packed_s8_interleaved_->create_ker();
    kernel_block4x8_mmla_packed_s8_interleaved_->create_ker();
    kernel_block8x8_mmla_packed_s8_->create_ker();
    kernel_block8x12_mmla_packed_s8_->create_ker();
    kernel_block4x16_mmla_packed_s8_interleaved_->create_ker();
    kernel_block4x4_udot_->create_ker();
    kernel_block4x4_dot_packed_s8_->create_ker();
    kernel_block4x4_mmla_packed_u8_->create_ker();
    kernel_block4x8_mmla_packed_u8_->create_ker();
    kernel_block4x16_mmla_packed_u8_->create_ker();
    kernel_block4x4_mmla_packed_u8_interleaved_->create_ker();
    kernel_block4x8_mmla_packed_u8_interleaved_->create_ker();
    kernel_block8x8_mmla_packed_u8_->create_ker();
    kernel_block8x12_mmla_packed_u8_->create_ker();
    kernel_block4x16_mmla_packed_u8_interleaved_->create_ker();
    kernel_block4x4_udot_packed_->create_ker();
}

bool BrgemmInt8ConvExecutor::supports(const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) {
    const auto& src_desc = config.descs.at(ARG_SRC);
    const auto& wei_desc = config.descs.at(ARG_WEI);
    const auto& dst_desc = config.descs.at(ARG_DST);
    const auto& src_dims = src_desc->getShape().getStaticDims();
    const auto& dst_dims = dst_desc->getShape().getStaticDims();
    const auto& wei_dims = wei_desc->getShape().getStaticDims();
    const auto src_prec = src_desc->getPrecision();
    const auto wei_prec = wei_desc->getPrecision();
    const auto dst_prec = dst_desc->getPrecision();
    const auto& attrs = config.attrs;

    DEBUG_LOG("BrgemmInt8ConvExecutor::supports src=", src_desc->serializeFormat(),
              " dst=", dst_desc->serializeFormat(),
              " wei=", wei_desc->serializeFormat(),
              " src_prec=", src_prec.to_string(),
              " wei_prec=", wei_prec.to_string(),
              " dst_prec=", dst_prec.to_string(),
              " src_dims=", src_dims,
              " dst_dims=", dst_dims,
              " wei_dims=", wei_dims,
              " stride=", attrs.stride,
              " dilation=", attrs.dilation,
              " padL=", attrs.paddingL,
              " padR=", attrs.paddingR,
              " isGrouped=", attrs.isGrouped,
              " postOps=", postOpsToString(attrs.postOps),
              " dqScales=", attrs.dqScales.size());
    jit_int8_conv_debug("supports src=", src_desc->serializeFormat(),
                        " dst=", dst_desc->serializeFormat(),
                        " wei=", wei_desc->serializeFormat(),
                        " src_prec=", src_prec.to_string(),
                        " wei_prec=", wei_prec.to_string(),
                        " dst_prec=", dst_prec.to_string(),
                        " src_dims=", src_dims,
                        " dst_dims=", dst_dims,
                        " wei_dims=", wei_dims,
                        " stride=", attrs.stride,
                        " dilation=", attrs.dilation,
                        " padL=", attrs.paddingL,
                        " padR=", attrs.paddingR,
                        " isGrouped=", attrs.isGrouped,
                        " postOps=", postOpsToString(attrs.postOps),
                        " dqScales=", attrs.dqScales.size());

    auto reject = [&](const char* reason, auto&&... extra) {
        jit_int8_conv_debug("reject: ", reason, std::forward<decltype(extra)>(extra)...);
        DEBUG_LOG(reason, std::forward<decltype(extra)>(extra)...);
        return false;
    };

    if (!MatchesMemoryFormatFilter(config.descs,
                                   std::vector<LayoutType>{LayoutType::nspc,
                                                           LayoutType::ncsp,
                                                           LayoutType::ncsp,
                                                           LayoutType::nspc},
                                   memoryFormatFilter,
                                   kConvMappingNotation)) {
        return reject(MEMORY_FORMAT_MISMATCH, " mapping filter mismatch");
    }

    if (!(src_dims.size() == 4 && dst_dims.size() == 4 && wei_dims.size() == 4)) {
        return reject(UNSUPPORTED_BY_EXECUTOR, " ranks src/dst/wei=", src_dims.size(), "/", dst_dims.size(), "/",
                      wei_dims.size());
    }
    if (attrs.stride.size() != 2) {
        return reject(UNSUPPORTED_BY_EXECUTOR, " stride_size=", attrs.stride.size());
    }
    if (attrs.dilation.size() != 2) {
        return reject(UNSUPPORTED_BY_EXECUTOR, " dilation_size=", attrs.dilation.size());
    }
    if (!(attrs.paddingL.size() == 2 && attrs.paddingR.size() == 2)) {
        return reject(UNSUPPORTED_BY_EXECUTOR, " paddingL_size=", attrs.paddingL.size(),
                      " paddingR_size=", attrs.paddingR.size());
    }
    if (attrs.isGrouped) {
        return reject(UNSUPPORTED_BY_EXECUTOR, " grouped convolution");
    }
    if (!is_zero(attrs.dilation)) {
        return reject(UNSUPPORTED_BY_EXECUTOR, " dilation=", attrs.dilation);
    }
    if (!is_non_negative(attrs.paddingL) || !is_non_negative(attrs.paddingR)) {
        return reject(UNSUPPORTED_BY_EXECUTOR, " paddingL/paddingR negative");
    }
    const auto* fq = getFqPostOp(attrs.postOps);
    if (!attrs.postOps.empty()) {
        if (!(attrs.postOps.size() == 1U && fq != nullptr)) {
            return reject(UNSUPPORTED_POST_OPS, " postOps=", postOpsToString(attrs.postOps));
        }
    }

    if (!any_of(src_prec, ov::element::u8, ov::element::i8)) {
        return reject(UNSUPPORTED_SRC_PRECISIONS, " src_prec=", src_prec.to_string());
    }
    if (wei_prec != ov::element::i8) {
        return reject(UNSUPPORTED_WEI_PRECISIONS, " wei_prec=", wei_prec.to_string());
    }
    const bool dst_i32 = dst_prec == ov::element::i32;
    const bool dst_i8 = any_of(dst_prec, ov::element::u8, ov::element::i8);
    if (fq) {
        if (!dst_i8) {
            return reject(UNSUPPORTED_DST_PRECISIONS, " dst_prec=", dst_prec.to_string());
        }
    } else {
        if (!(dst_i32 || dst_i8)) {
            return reject(UNSUPPORTED_DST_PRECISIONS, " dst_prec=", dst_prec.to_string());
        }
        if (dst_i8) {
            if (!attrs.postOps.empty()) {
                return reject(UNSUPPORTED_POST_OPS, " postOps=", postOpsToString(attrs.postOps));
            }
        }
    }
    if (attrs.withBias) {
        const auto bias_prec = config.descs.at(ARG_BIAS)->getPrecision();
        const bool bias_i32 = bias_prec == ov::element::i32;
        const bool bias_fp = any_of(bias_prec, ov::element::f32, ov::element::f16);
        if (!bias_i32 && !bias_fp) {
            return reject(UNSUPPORTED_BIAS_PRECISIONS, " bias_prec=", bias_prec.to_string());
        }
        if (bias_fp) {
            if (dst_i32) {
                return reject(UNSUPPORTED_BIAS_PRECISIONS, " bias_prec=", bias_prec.to_string(),
                              " dst_prec=", dst_prec.to_string());
            }
            if (!fq && attrs.dqScales.empty()) {
                return reject(UNSUPPORTED_POST_OPS, " float bias requires dqScales or FakeQuantize");
            }
        }
    }

    const auto& wei_shape = wei_dims;
    const size_t wei_rank = wei_shape.size();
    const size_t kh_idx = wei_rank - 2;
    const size_t kw_idx = wei_rank - 1;
    const size_t KH = wei_shape[kh_idx];
    const size_t KW = wei_shape[kw_idx];
    if (!((KH == 1 && KW == 1) || (KH == 3 && KW == 3) || (KH == 5 && KW == 5))) {
        return reject(UNSUPPORTED_BY_EXECUTOR, " kernel=", KH, "x", KW);
    }
    if (fq) {
        if (!isSupportedFq(*fq, dst_dims[1])) {
            return reject(UNSUPPORTED_POST_OPS, " fq unsupported, oc=", dst_dims[1]);
        }
        const auto& dq_scales = attrs.dqScales;
        if (!dq_scales.empty()) {
            if (!(dq_scales.size() == 1 || dq_scales.size() == dst_dims[1])) {
                return reject(UNSUPPORTED_POST_OPS, " dqScales=", dq_scales.size(), " oc=", dst_dims[1]);
            }
        }
    } else if (dst_i8) {
        const auto& dq_scales = attrs.dqScales;
        if (!dq_scales.empty()) {
            if (!(dq_scales.size() == 1 || dq_scales.size() == dst_dims[1])) {
                return reject(UNSUPPORTED_POST_OPS, " dqScales=", dq_scales.size(), " oc=", dst_dims[1]);
            }
        }
    }

    return true;
}

bool BrgemmInt8ConvExecutor::update([[maybe_unused]] const MemoryArgs& memory) {
    bias_prec_ = ov::element::dynamic;
    bias_f32_.clear();
    bias_comp_1x1_.clear();
    if (attrs_.withBias) {
        auto bit = memory.find(ARG_BIAS);
        if (bit != memory.end()) {
            bias_prec_ = bit->second->getDescPtr()->getPrecision();
            if (any_of(bias_prec_, ov::element::f32, ov::element::f16)) {
                PlainTensor bias(bit->second);
                const auto& bias_dims = bias.shape();
                if (!bias_dims.empty()) {
                    const size_t OC = bias_dims[0];
                    bias_f32_.resize(OC);
                    if (bias_prec_ == ov::element::f16) {
                        const ov::float16* src = bias.ptr<ov::float16>(0);
                        for (size_t oc = 0; oc < OC; ++oc) {
                            bias_f32_[oc] = static_cast<float>(src[oc]);
                        }
                    } else {
                        const float* src = bias.ptr<float>(0);
                        std::copy(src, src + OC, bias_f32_.begin());
                    }
                }
            }
        }
    }

    auto it = memory.find(ARG_WEI);
    if (it == memory.end()) {
        return true;
    }

    PlainTensor wei(it->second);
    const auto& wei_dims = wei.shape();
    if (wei_dims.size() < 4) {
        return true;
    }

    const size_t OC = wei_dims[0];
    const size_t IC = wei_dims[1];
    const size_t KH = wei_dims[wei_dims.size() - 2];
    const size_t KW = wei_dims[wei_dims.size() - 1];

    if (KH == 1 && KW == 1) {
        const void* wei_ptr_base = wei.ptr<int8_t>(0, 0, 0, 0);
        if (packed_wei_1x1_src_ != wei_ptr_base || packed_wei_1x1_oc_ != OC || packed_wei_1x1_ic_ != IC) {
            const bool prefer_sve_int8 = ov::intel_cpu::isSVEInt8Isa(runtime_isa_);
            packed_wei_1x1_.assign(IC * OC, 0);
            packed_wei_1x1_col_.assign(OC * IC, 0);
            packed_wei_1x1_dot4_.clear();
            packed_wei_1x1_dot8_.clear();
            packed_wei_1x1_dot8_interleaved_.clear();
            packed_wei_1x1_dot16_interleaved_.clear();
            packed_wei_1x1_dot32_interleaved_.clear();
            packed_wei_1x1_mmla4_.clear();
            packed_wei_1x1_mmla8_.clear();
            packed_wei_1x1_mmla12_.clear();
            packed_wei_1x1_mmla16_.clear();
            packed_wei_1x1_dot4_stride_ = 0;
            packed_wei_1x1_dot8_stride_ = 0;
            packed_wei_1x1_dot8_interleaved_stride_ = 0;
            packed_wei_1x1_dot16_interleaved_stride_ = 0;
            packed_wei_1x1_dot32_interleaved_stride_ = 0;
            packed_wei_1x1_mmla4_stride_ = 0;
            packed_wei_1x1_mmla8_stride_ = 0;
            packed_wei_1x1_mmla12_stride_ = 0;
            packed_wei_1x1_mmla16_stride_ = 0;
            wei_comp_1x1_.assign(OC, 0);
            std::vector<int64_t> comp_accum(OC, 0);
            for (size_t ic = 0; ic < IC; ++ic) {
                for (size_t oc = 0; oc < OC; ++oc) {
                    const int8_t v = *wei.ptr<int8_t>(oc, ic, 0, 0);
                    packed_wei_1x1_[ic * OC + oc] = v;
                    packed_wei_1x1_col_[oc * IC + ic] = v;
                    comp_accum[oc] += static_cast<int64_t>(v);
                }
            }
            for (size_t oc = 0; oc < OC; ++oc) {
                wei_comp_1x1_[oc] = static_cast<int32_t>(comp_accum[oc] * 128);
            }
            if (!prefer_sve_int8 && has_dotprod_) {
                const size_t oc_blocks16 = OC / 16;
                const size_t oc_blocks32 = OC / 32;
                const size_t oc_blocks8 = OC / 8;
                const size_t oc_blocks4 = OC / 4;
                packed_wei_1x1_dot8_stride_ = packed_block_stride(IC, 8);
                packed_wei_1x1_dot4_stride_ = packed_block_stride(IC, 4);
                if (IC % 4 == 0) {
                    packed_wei_1x1_dot8_interleaved_stride_ = packed_block_stride_dot_interleaved(IC, 8);
                    packed_wei_1x1_dot16_interleaved_stride_ = packed_block_stride_dot_interleaved(IC, 16);
                    packed_wei_1x1_dot32_interleaved_stride_ = packed_block_stride_dot_interleaved(IC, 32);
                }
                if (oc_blocks32 > 0 && packed_wei_1x1_dot32_interleaved_stride_ > 0) {
                    packed_wei_1x1_dot32_interleaved_.assign(
                        oc_blocks32 * packed_wei_1x1_dot32_interleaved_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks32; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 32 * IC;
                        int8_t* dst_block = packed_wei_1x1_dot32_interleaved_.data() +
                                            ocb * packed_wei_1x1_dot32_interleaved_stride_;
                        pack_dot_block_interleaved4(src_block, IC, 32, dst_block);
                    }
                }
                if (oc_blocks16 > 0 && packed_wei_1x1_dot16_interleaved_stride_ > 0) {
                    packed_wei_1x1_dot16_interleaved_.assign(
                        oc_blocks16 * packed_wei_1x1_dot16_interleaved_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks16; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 16 * IC;
                        int8_t* dst_block = packed_wei_1x1_dot16_interleaved_.data() +
                                            ocb * packed_wei_1x1_dot16_interleaved_stride_;
                        pack_dot_block_interleaved4(src_block, IC, 16, dst_block);
                    }
                }
                if (oc_blocks8 > 0) {
                    packed_wei_1x1_dot8_.assign(oc_blocks8 * packed_wei_1x1_dot8_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 8 * IC;
                        int8_t* dst_block = packed_wei_1x1_dot8_.data() + ocb * packed_wei_1x1_dot8_stride_;
                        pack_dot_block(src_block, IC, 8, dst_block);
                    }
                }
                if (oc_blocks8 > 0 && packed_wei_1x1_dot8_interleaved_stride_ > 0) {
                    packed_wei_1x1_dot8_interleaved_.assign(oc_blocks8 * packed_wei_1x1_dot8_interleaved_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 8 * IC;
                        int8_t* dst_block =
                            packed_wei_1x1_dot8_interleaved_.data() + ocb * packed_wei_1x1_dot8_interleaved_stride_;
                        pack_dot_block_interleaved4(src_block, IC, 8, dst_block);
                    }
                }
                if (oc_blocks4 > 0) {
                    packed_wei_1x1_dot4_.assign(oc_blocks4 * packed_wei_1x1_dot4_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks4; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 4 * IC;
                        int8_t* dst_block = packed_wei_1x1_dot4_.data() + ocb * packed_wei_1x1_dot4_stride_;
                        pack_dot_block(src_block, IC, 4, dst_block);
                    }
                }
            }
            if (!prefer_sve_int8 && has_i8mm_ && (IC % 8 == 0)) {
                const size_t ic_padded = round_up(IC, 8);
                const size_t oc_blocks16 = OC / 16;
                const size_t oc_blocks12 = OC / 12;
                const size_t oc_blocks8 = OC / 8;
                const size_t oc_blocks4 = OC / 4;
                packed_wei_1x1_mmla16_stride_ = packed_block_stride_mmla(IC, 16);
                if (oc_blocks16 > 0 && packed_wei_1x1_mmla16_stride_ > 0) {
                    packed_wei_1x1_mmla16_.assign(oc_blocks16 * packed_wei_1x1_mmla16_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks16; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 16 * IC;
                        int8_t* dst_block =
                            packed_wei_1x1_mmla16_.data() + ocb * packed_wei_1x1_mmla16_stride_;
                        pack_mmla_block(src_block, IC, ic_padded, 16, dst_block);
                    }
                }
                packed_wei_1x1_mmla8_stride_ = packed_block_stride_mmla(IC, 8);
                if (oc_blocks8 > 0 && packed_wei_1x1_mmla8_stride_ > 0) {
                    packed_wei_1x1_mmla8_.assign(oc_blocks8 * packed_wei_1x1_mmla8_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 8 * IC;
                        int8_t* dst_block =
                            packed_wei_1x1_mmla8_.data() + ocb * packed_wei_1x1_mmla8_stride_;
                        pack_mmla_block(src_block, IC, ic_padded, 8, dst_block);
                    }
                }
                packed_wei_1x1_mmla12_stride_ = packed_block_stride_mmla(IC, 12);
                if (oc_blocks12 > 0 && packed_wei_1x1_mmla12_stride_ > 0) {
                    packed_wei_1x1_mmla12_.assign(oc_blocks12 * packed_wei_1x1_mmla12_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks12; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 12 * IC;
                        int8_t* dst_block =
                            packed_wei_1x1_mmla12_.data() + ocb * packed_wei_1x1_mmla12_stride_;
                        pack_mmla_block(src_block, IC, ic_padded, 12, dst_block);
                    }
                }
                packed_wei_1x1_mmla4_stride_ = packed_block_stride_mmla(IC, 4);
                if (oc_blocks4 > 0 && packed_wei_1x1_mmla4_stride_ > 0) {
                    packed_wei_1x1_mmla4_.assign(oc_blocks4 * packed_wei_1x1_mmla4_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks4; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 4 * IC;
                        int8_t* dst_block =
                            packed_wei_1x1_mmla4_.data() + ocb * packed_wei_1x1_mmla4_stride_;
                        pack_mmla_block(src_block, IC, ic_padded, 4, dst_block);
                    }
                }
            }
            packed_wei_1x1_src_ = wei_ptr_base;
            packed_wei_1x1_oc_ = OC;
            packed_wei_1x1_ic_ = IC;
        }
        if (!wei_comp_1x1_.empty()) {
            const bool bias_i32 = bias_prec_ == ov::element::i32;
            if (attrs_.withBias && bias_i32) {
                auto bit = memory.find(ARG_BIAS);
                if (bit != memory.end()) {
                    PlainTensor bias(bit->second);
                    const auto& bias_dims = bias.shape();
                    if (!bias_dims.empty()) {
                        const size_t bias_oc = bias_dims[0];
                        bias_comp_1x1_.resize(bias_oc);
                        const int32_t* bias_ptr = bias.ptr<int32_t>(0);
                        for (size_t oc = 0; oc < bias_oc; ++oc) {
                            bias_comp_1x1_[oc] = bias_ptr[oc] + wei_comp_1x1_[oc];
                        }
                    }
                }
            } else if (!attrs_.withBias) {
                bias_comp_1x1_ = wei_comp_1x1_;
            }
        }
    } else {
        packed_wei_1x1_.clear();
        packed_wei_1x1_col_.clear();
        packed_wei_1x1_dot4_.clear();
        packed_wei_1x1_dot8_.clear();
        packed_wei_1x1_dot8_interleaved_.clear();
        packed_wei_1x1_dot16_interleaved_.clear();
        packed_wei_1x1_dot32_interleaved_.clear();
        packed_wei_1x1_mmla4_.clear();
        packed_wei_1x1_mmla8_.clear();
        packed_wei_1x1_mmla12_.clear();
        packed_wei_1x1_mmla16_.clear();
        packed_wei_1x1_dot4_stride_ = 0;
        packed_wei_1x1_dot8_stride_ = 0;
        packed_wei_1x1_dot8_interleaved_stride_ = 0;
        packed_wei_1x1_dot16_interleaved_stride_ = 0;
        packed_wei_1x1_dot32_interleaved_stride_ = 0;
        packed_wei_1x1_mmla4_stride_ = 0;
        packed_wei_1x1_mmla8_stride_ = 0;
        packed_wei_1x1_mmla12_stride_ = 0;
        packed_wei_1x1_mmla16_stride_ = 0;
        wei_comp_1x1_.clear();
        bias_comp_1x1_.clear();
        packed_wei_1x1_src_ = nullptr;
        packed_wei_1x1_oc_ = 0;
        packed_wei_1x1_ic_ = 0;
    }

    if (KH != 1 || KW != 1) {
        const void* wei_ptr_base = wei.ptr<int8_t>(0, 0, 0, 0);
        if (packed_wei_brgemm_src_ != wei_ptr_base || packed_wei_brgemm_oc_ != OC || packed_wei_brgemm_ic_ != IC ||
            packed_wei_brgemm_kh_ != KH || packed_wei_brgemm_kw_ != KW) {
            const bool prefer_sve_int8 = ov::intel_cpu::isSVEInt8Isa(runtime_isa_);
            packed_wei_brgemm_.assign(KH * KW * IC * OC, 0);
            packed_wei_brgemm_col_.assign(KH * KW * OC * IC, 0);
            packed_wei_brgemm_dot4_.clear();
            packed_wei_brgemm_dot8_.clear();
            packed_wei_brgemm_mmla4_.clear();
            packed_wei_brgemm_mmla8_.clear();
            packed_wei_brgemm_mmla16_.clear();
            packed_wei_brgemm_mmla4_fused_.clear();
            packed_wei_brgemm_mmla8_fused_.clear();
            packed_wei_brgemm_mmla12_fused_.clear();
            packed_wei_brgemm_mmla16_fused_.clear();
            packed_wei_brgemm_dot4_stride_ = 0;
            packed_wei_brgemm_dot8_stride_ = 0;
            packed_wei_brgemm_mmla4_stride_ = 0;
            packed_wei_brgemm_mmla8_stride_ = 0;
            packed_wei_brgemm_mmla16_stride_ = 0;
            packed_wei_brgemm_mmla4_fused_stride_ = 0;
            packed_wei_brgemm_mmla8_fused_stride_ = 0;
            packed_wei_brgemm_mmla12_fused_stride_ = 0;
            packed_wei_brgemm_mmla16_fused_stride_ = 0;
            packed_wei_brgemm_mmla_fused_k_ = 0;
            wei_comp_brgemm_.assign(OC, 0);
            std::vector<int64_t> comp_accum(OC, 0);
            for (size_t kh = 0; kh < KH; ++kh) {
                for (size_t kw = 0; kw < KW; ++kw) {
                    for (size_t ic = 0; ic < IC; ++ic) {
                        for (size_t oc = 0; oc < OC; ++oc) {
                            const int8_t v = *wei.ptr<int8_t>(oc, ic, kh, kw);
                            const size_t dst_idx = ((kh * KW + kw) * IC + ic) * OC + oc;
                            const size_t dst_col_idx = ((kh * KW + kw) * OC + oc) * IC + ic;
                            packed_wei_brgemm_[dst_idx] = v;
                            packed_wei_brgemm_col_[dst_col_idx] = v;
                            comp_accum[oc] += static_cast<int64_t>(v);
                        }
                    }
                }
            }
            for (size_t oc = 0; oc < OC; ++oc) {
                wei_comp_brgemm_[oc] = static_cast<int32_t>(comp_accum[oc] * 128);
            }
            if (!prefer_sve_int8 && has_dotprod_) {
                const size_t spatial = KH * KW;
                const size_t oc_blocks8 = OC / 8;
                const size_t oc_blocks4 = OC / 4;
                packed_wei_brgemm_dot8_stride_ = packed_block_stride(IC, 8);
                packed_wei_brgemm_dot4_stride_ = packed_block_stride(IC, 4);
                if (oc_blocks8 > 0) {
                    packed_wei_brgemm_dot8_.assign(spatial * oc_blocks8 * packed_wei_brgemm_dot8_stride_, 0);
                    for (size_t kh = 0; kh < KH; ++kh) {
                        for (size_t kw = 0; kw < KW; ++kw) {
                            const size_t khkw = kh * KW + kw;
                            for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
                                const int8_t* src_block =
                                    packed_wei_brgemm_col_.data() + (khkw * OC + ocb * 8) * IC;
                                int8_t* dst_block = packed_wei_brgemm_dot8_.data() +
                                                    (khkw * oc_blocks8 + ocb) * packed_wei_brgemm_dot8_stride_;
                                pack_dot_block(src_block, IC, 8, dst_block);
                            }
                        }
                    }
                }
                if (oc_blocks4 > 0) {
                    packed_wei_brgemm_dot4_.assign(spatial * oc_blocks4 * packed_wei_brgemm_dot4_stride_, 0);
                    for (size_t kh = 0; kh < KH; ++kh) {
                        for (size_t kw = 0; kw < KW; ++kw) {
                            const size_t khkw = kh * KW + kw;
                            for (size_t ocb = 0; ocb < oc_blocks4; ++ocb) {
                                const int8_t* src_block =
                                    packed_wei_brgemm_col_.data() + (khkw * OC + ocb * 4) * IC;
                                int8_t* dst_block = packed_wei_brgemm_dot4_.data() +
                                                    (khkw * oc_blocks4 + ocb) * packed_wei_brgemm_dot4_stride_;
                                pack_dot_block(src_block, IC, 4, dst_block);
                            }
                        }
                    }
                }
            }
            if (!prefer_sve_int8 && has_i8mm_ && (IC % 8 == 0)) {
                const size_t ic_padded = round_up(IC, 8);
                const size_t spatial = KH * KW;
                const size_t oc_blocks16 = OC / 16;
                const size_t oc_blocks8 = OC / 8;
                const size_t oc_blocks4 = OC / 4;
                packed_wei_brgemm_mmla16_stride_ = packed_block_stride_mmla(IC, 16);
                if (oc_blocks16 > 0 && packed_wei_brgemm_mmla16_stride_ > 0) {
                    packed_wei_brgemm_mmla16_.assign(spatial * oc_blocks16 * packed_wei_brgemm_mmla16_stride_, 0);
                    for (size_t kh = 0; kh < KH; ++kh) {
                        for (size_t kw = 0; kw < KW; ++kw) {
                            const size_t khkw = kh * KW + kw;
                            for (size_t ocb = 0; ocb < oc_blocks16; ++ocb) {
                                const int8_t* src_block =
                                    packed_wei_brgemm_col_.data() + (khkw * OC + ocb * 16) * IC;
                                int8_t* dst_block = packed_wei_brgemm_mmla16_.data() +
                                                    (khkw * oc_blocks16 + ocb) * packed_wei_brgemm_mmla16_stride_;
                                pack_mmla_block(src_block, IC, ic_padded, 16, dst_block);
                            }
                        }
                    }
                }
                packed_wei_brgemm_mmla8_stride_ = packed_block_stride_mmla(IC, 8);
                if (oc_blocks8 > 0 && packed_wei_brgemm_mmla8_stride_ > 0) {
                    packed_wei_brgemm_mmla8_.assign(spatial * oc_blocks8 * packed_wei_brgemm_mmla8_stride_, 0);
                    for (size_t kh = 0; kh < KH; ++kh) {
                        for (size_t kw = 0; kw < KW; ++kw) {
                            const size_t khkw = kh * KW + kw;
                            for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
                                const int8_t* src_block =
                                    packed_wei_brgemm_col_.data() + (khkw * OC + ocb * 8) * IC;
                                int8_t* dst_block = packed_wei_brgemm_mmla8_.data() +
                                                    (khkw * oc_blocks8 + ocb) * packed_wei_brgemm_mmla8_stride_;
                                pack_mmla_block(src_block, IC, ic_padded, 8, dst_block);
                            }
                        }
                    }
                }
                packed_wei_brgemm_mmla4_stride_ = packed_block_stride_mmla(IC, 4);
                if (oc_blocks4 > 0 && packed_wei_brgemm_mmla4_stride_ > 0) {
                    packed_wei_brgemm_mmla4_.assign(spatial * oc_blocks4 * packed_wei_brgemm_mmla4_stride_, 0);
                    for (size_t kh = 0; kh < KH; ++kh) {
                        for (size_t kw = 0; kw < KW; ++kw) {
                            const size_t khkw = kh * KW + kw;
                            for (size_t ocb = 0; ocb < oc_blocks4; ++ocb) {
                                const int8_t* src_block =
                                    packed_wei_brgemm_col_.data() + (khkw * OC + ocb * 4) * IC;
                                int8_t* dst_block = packed_wei_brgemm_mmla4_.data() +
                                                    (khkw * oc_blocks4 + ocb) * packed_wei_brgemm_mmla4_stride_;
                                pack_mmla_block(src_block, IC, ic_padded, 4, dst_block);
                            }
                        }
                    }
                }
            }
            if (!prefer_sve_int8 && has_i8mm_) {
                const size_t fused_k = IC * KH * KW;
                const size_t fused_kp = round_up(fused_k, 8);
                packed_wei_brgemm_mmla_fused_k_ = fused_kp;
                std::vector<int8_t> fused_col(OC * fused_k);
                for (size_t oc = 0; oc < OC; ++oc) {
                    size_t dst_idx = oc * fused_k;
                    for (size_t kh = 0; kh < KH; ++kh) {
                        for (size_t kw = 0; kw < KW; ++kw) {
                            for (size_t ic = 0; ic < IC; ++ic) {
                                fused_col[dst_idx++] = *wei.ptr<int8_t>(oc, ic, kh, kw);
                            }
                        }
                    }
                }
                const size_t oc_blocks16 = OC / 16;
                const size_t oc_blocks12 = OC / 12;
                const size_t oc_blocks8 = OC / 8;
                const size_t oc_blocks4 = OC / 4;
                packed_wei_brgemm_mmla16_fused_stride_ = packed_block_stride_mmla(fused_k, 16);
                if (oc_blocks16 > 0 && packed_wei_brgemm_mmla16_fused_stride_ > 0) {
                    packed_wei_brgemm_mmla16_fused_.assign(oc_blocks16 * packed_wei_brgemm_mmla16_fused_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks16; ++ocb) {
                        const int8_t* src_block = fused_col.data() + ocb * 16 * fused_k;
                        int8_t* dst_block =
                            packed_wei_brgemm_mmla16_fused_.data() + ocb * packed_wei_brgemm_mmla16_fused_stride_;
                        pack_mmla_block(src_block, fused_k, fused_kp, 16, dst_block);
                    }
                }
                packed_wei_brgemm_mmla8_fused_stride_ = packed_block_stride_mmla(fused_k, 8);
                if (oc_blocks8 > 0 && packed_wei_brgemm_mmla8_fused_stride_ > 0) {
                    packed_wei_brgemm_mmla8_fused_.assign(oc_blocks8 * packed_wei_brgemm_mmla8_fused_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
                        const int8_t* src_block = fused_col.data() + ocb * 8 * fused_k;
                        int8_t* dst_block =
                            packed_wei_brgemm_mmla8_fused_.data() + ocb * packed_wei_brgemm_mmla8_fused_stride_;
                        pack_mmla_block(src_block, fused_k, fused_kp, 8, dst_block);
                    }
                }
                packed_wei_brgemm_mmla12_fused_stride_ = packed_block_stride_mmla(fused_k, 12);
                if (oc_blocks12 > 0 && packed_wei_brgemm_mmla12_fused_stride_ > 0) {
                    packed_wei_brgemm_mmla12_fused_.assign(oc_blocks12 * packed_wei_brgemm_mmla12_fused_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks12; ++ocb) {
                        const int8_t* src_block = fused_col.data() + ocb * 12 * fused_k;
                        int8_t* dst_block =
                            packed_wei_brgemm_mmla12_fused_.data() + ocb * packed_wei_brgemm_mmla12_fused_stride_;
                        pack_mmla_block(src_block, fused_k, fused_kp, 12, dst_block);
                    }
                }
                packed_wei_brgemm_mmla4_fused_stride_ = packed_block_stride_mmla(fused_k, 4);
                if (oc_blocks4 > 0 && packed_wei_brgemm_mmla4_fused_stride_ > 0) {
                    packed_wei_brgemm_mmla4_fused_.assign(oc_blocks4 * packed_wei_brgemm_mmla4_fused_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks4; ++ocb) {
                        const int8_t* src_block = fused_col.data() + ocb * 4 * fused_k;
                        int8_t* dst_block =
                            packed_wei_brgemm_mmla4_fused_.data() + ocb * packed_wei_brgemm_mmla4_fused_stride_;
                        pack_mmla_block(src_block, fused_k, fused_kp, 4, dst_block);
                    }
                }
            }
            packed_wei_brgemm_src_ = wei_ptr_base;
            packed_wei_brgemm_oc_ = OC;
            packed_wei_brgemm_ic_ = IC;
            packed_wei_brgemm_kh_ = KH;
            packed_wei_brgemm_kw_ = KW;
        }
    } else {
        packed_wei_brgemm_.clear();
        packed_wei_brgemm_col_.clear();
        packed_wei_brgemm_dot4_.clear();
        packed_wei_brgemm_dot8_.clear();
        packed_wei_brgemm_mmla4_.clear();
        packed_wei_brgemm_mmla8_.clear();
        packed_wei_brgemm_mmla16_.clear();
        packed_wei_brgemm_mmla4_fused_.clear();
        packed_wei_brgemm_mmla8_fused_.clear();
        packed_wei_brgemm_mmla12_fused_.clear();
        packed_wei_brgemm_mmla16_fused_.clear();
        packed_wei_brgemm_dot4_stride_ = 0;
        packed_wei_brgemm_dot8_stride_ = 0;
        packed_wei_brgemm_mmla4_stride_ = 0;
        packed_wei_brgemm_mmla8_stride_ = 0;
        packed_wei_brgemm_mmla16_stride_ = 0;
        packed_wei_brgemm_mmla4_fused_stride_ = 0;
        packed_wei_brgemm_mmla8_fused_stride_ = 0;
        packed_wei_brgemm_mmla12_fused_stride_ = 0;
        packed_wei_brgemm_mmla16_fused_stride_ = 0;
        packed_wei_brgemm_mmla_fused_k_ = 0;
        wei_comp_brgemm_.clear();
        packed_wei_brgemm_src_ = nullptr;
        packed_wei_brgemm_oc_ = 0;
        packed_wei_brgemm_ic_ = 0;
        packed_wei_brgemm_kh_ = 0;
        packed_wei_brgemm_kw_ = 0;
    }

    if (KH != 1 || KW != 1) {
        const void* wei_ptr_base = wei.ptr<int8_t>(0, 0, 0, 0);
        if (packed_wei_src_ != wei_ptr_base || packed_oc_ != OC || packed_ic_ != IC || packed_kh_ != KH ||
            packed_kw_ != KW) {
            packed_wei_.assign(KH * KW * OC * IC, 0);
            for (size_t kh = 0; kh < KH; ++kh) {
                for (size_t kw = 0; kw < KW; ++kw) {
                    for (size_t oc = 0; oc < OC; ++oc) {
                        for (size_t ic = 0; ic < IC; ++ic) {
                            const size_t dst_idx = ((kh * KW + kw) * OC + oc) * IC + ic;
                            packed_wei_[dst_idx] = *wei.ptr<int8_t>(oc, ic, kh, kw);
                        }
                    }
                }
            }
            packed_wei_src_ = wei_ptr_base;
            packed_oc_ = OC;
            packed_ic_ = IC;
            packed_kh_ = KH;
            packed_kw_ = KW;
        }
    } else {
        packed_wei_.clear();
        packed_wei_src_ = nullptr;
        packed_oc_ = 0;
        packed_ic_ = 0;
        packed_kh_ = 0;
        packed_kw_ = 0;
    }
    return true;
}

bool BrgemmInt8ConvExecutor::execute_brgemm_1x1(PlainTensor& dst, const MemoryArgs& memory) {
    PlainTensor src(memory.at(ARG_SRC));
    PlainTensor wei(memory.at(ARG_WEI));

    const auto& src_dims = src.shape();
    const auto& dst_dims = dst.shape();
    const auto& wei_dims = wei.shape();
    if (src_dims.size() != 4 || dst_dims.size() != 4 || wei_dims.size() != 4) {
        return false;
    }

    const size_t KH = wei_dims[wei_dims.size() - 2];
    const size_t KW = wei_dims[wei_dims.size() - 1];
    if (KH != 1 || KW != 1) {
        return false;
    }

    if (attrs_.stride.size() != 2 || attrs_.paddingL.size() != 2 || attrs_.paddingR.size() != 2) {
        return false;
    }
    if (!is_zero(attrs_.dilation)) {
        return false;
    }
    if (packed_wei_1x1_.empty()) {
        return false;
    }

    const bool prefer_sve_int8 = ov::intel_cpu::isSVEInt8Isa(runtime_isa_);
    const bool prefer_mmla_1x1 =
        !prefer_sve_int8 && has_i8mm_ && (src_dims[1] % 8 == 0) &&
        (!packed_wei_1x1_mmla16_.empty() || !packed_wei_1x1_mmla8_.empty() || !packed_wei_1x1_mmla4_.empty());
    const size_t N = src_dims[0];
    const size_t IC = src_dims[1];
    const size_t IH = src_dims[2];
    const size_t IW = src_dims[3];
    const size_t OH = dst_dims[2];
    const size_t OW = dst_dims[3];
    const size_t OC = dst_dims[1];
    if (!prefer_sve_int8 && (has_dotprod_ || has_i8mm_)) {
        return false;
    }

    if (OW < brgemm_1x1_m_blk_) {
        return false;
    }
    if (prefer_mmla_1x1 && OW >= 16) {
        return false;
    }

    const size_t stride_h = attrs_.stride[0];
    const size_t stride_w = attrs_.stride[1];
    const ptrdiff_t pad_t = attrs_.paddingL[0];
    const ptrdiff_t pad_l = attrs_.paddingL[1];
    const ptrdiff_t pad_r = attrs_.paddingR[1];
    const bool src_signed = src.get_precision() == ov::element::i8;
    const bool use_dot_s8 = src_signed && has_dotprod_ && !prefer_sve_int8;
    const bool use_dot_u8 = !src_signed && has_dotprod_ && !prefer_sve_int8;
    const bool use_packed8_1x1 = has_dotprod_ && !packed_wei_1x1_dot8_.empty();
    const bool use_block2x16_interleaved_1x1 =
        (OC % 16 == 0) && (IC % 4 == 0) && !packed_wei_1x1_dot16_interleaved_.empty() && (use_dot_s8 || use_dot_u8);
    const bool use_block2x32_interleaved_1x1 =
        (OC % 32 == 0) && (IC % 4 == 0) && !packed_wei_1x1_dot32_interleaved_.empty() && (use_dot_s8 || use_dot_u8);
    const bool prefer_block2x8_1x1 = (OW < 16) && (stride_w == 1) && (pad_l == 0) && (pad_r == 0) &&
                                     use_packed8_1x1 && (OC % 8 == 0) && (use_dot_s8 || use_dot_u8);
    const bool prefer_block2x16_1x1 =
        (OW < 16) && (stride_w == 1) && (pad_l == 0) && (pad_r == 0) && use_block2x16_interleaved_1x1;
    const bool prefer_block2x32_1x1 =
        (OW < 16) && (stride_w == 1) && (pad_l == 0) && (pad_r == 0) && use_block2x32_interleaved_1x1;
    if (prefer_block2x8_1x1 || prefer_block2x16_1x1 || prefer_block2x32_1x1) {
        return false;
    }

    const size_t lda = src.stride(3) * stride_w;
    const size_t ldc = dst.stride(3);
    if (lda < IC || ldc < OC) {
        return false;
    }

    auto& brgemm_kernel = src_signed ? brgemm_1x1_s8_ : brgemm_1x1_u8_;
    if (!brgemm_kernel || brgemm_1x1_oc_ != OC || brgemm_1x1_ic_ != IC || brgemm_1x1_lda_ != lda ||
        brgemm_1x1_ldc_ != ldc) {
        brgemm_kernel = std::make_shared<BrgemmInt8Kernel>(brgemm_1x1_m_blk_, OC, IC, lda, OC, ldc, src_signed);
        brgemm_1x1_oc_ = OC;
        brgemm_1x1_ic_ = IC;
        brgemm_1x1_lda_ = lda;
        brgemm_1x1_ldc_ = ldc;
    }
    if (!logged_brgemm_1x1_path_) {
        const char* family =
            brgemm_kernel->uses_brgemm() ? "onednn_brgemm_sve" : fallback_family_name_for_debug(has_i8mm_, has_dotprod_);
        jit_int8_conv_debug("selected_path=brgemm_1x1",
                            " family=", family,
                            " src=", src_signed ? "s8" : "u8",
                            " Mblk=", brgemm_1x1_m_blk_,
                            " IC=", IC,
                            " OC=", OC,
                            " lda=", lda,
                            " ldc=", ldc);
        logged_brgemm_1x1_path_ = true;
    }
    const auto bias_prec = (bias_prec_ == ov::element::dynamic && attrs_.withBias)
                               ? memory.at(ARG_BIAS)->getDescPtr()->getPrecision()
                               : bias_prec_;
    const bool bias_i32 = bias_prec == ov::element::i32;
    const int32_t* bias_ptr = nullptr;
    PlainTensor bias;
    if (attrs_.withBias && bias_i32) {
        bias.reset(memory.at(ARG_BIAS));
        bias_ptr = bias.ptr<int32_t>(0);
    }
    const bool use_bias_comp_1x1 = !src_signed && !bias_comp_1x1_.empty();
    if (use_bias_comp_1x1) {
        bias_ptr = bias_comp_1x1_.data();
    }
    const bool use_comp = !src_signed && brgemm_kernel && !brgemm_kernel->uses_brgemm() && !wei_comp_1x1_.empty();
    const int32_t* comp_ptr = (use_comp && !use_bias_comp_1x1) ? wei_comp_1x1_.data() : nullptr;

    const auto ker = src_signed ? kernel_s8_->ker() : kernel_u8_->ker();
    const int8_t* packed_wei_1x1_ptr =
        brgemm_kernel->uses_brgemm() ? packed_wei_1x1_.data() : packed_wei_1x1_col_.data();
    for (size_t n = 0; n < N; ++n) {
        for (size_t oh = 0; oh < OH; ++oh) {
            const ptrdiff_t ih = static_cast<ptrdiff_t>(oh * stride_h) - pad_t;
            if (ih < 0 || ih >= static_cast<ptrdiff_t>(IH)) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    int32_t* dst_ptr = dst.ptr<int32_t>(n, 0, oh, ow);
                    if (bias_ptr) {
                        for (size_t oc = 0; oc < OC; ++oc) {
                            dst_ptr[oc] = bias_ptr[oc];
                        }
                    } else {
                        std::fill(dst_ptr, dst_ptr + OC, 0);
                    }
                }
                continue;
            }

            const ptrdiff_t iw_start = -pad_l;
            const ptrdiff_t iw_end = static_cast<ptrdiff_t>(IW - 1) + pad_r;
            const size_t ow_start = iw_start >= 0 ? 0 : static_cast<size_t>((-iw_start + stride_w - 1) / stride_w);
            const size_t ow_end = iw_end < 0
                                      ? 0
                                      : static_cast<size_t>(std::min<ptrdiff_t>(
                                            static_cast<ptrdiff_t>(OW - 1),
                                            iw_end / static_cast<ptrdiff_t>(stride_w)));

            for (size_t ow = 0; ow < ow_start && ow < OW; ++ow) {
                int32_t* dst_ptr = dst.ptr<int32_t>(n, 0, oh, ow);
                if (bias_ptr) {
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dst_ptr[oc] = bias_ptr[oc];
                    }
                } else {
                    std::fill(dst_ptr, dst_ptr + OC, 0);
                }
            }

            size_t ow = ow_start;
            if (ow_start <= ow_end && ow < OW) {
                const ptrdiff_t iw0 = static_cast<ptrdiff_t>(ow * stride_w) - pad_l;
                const uint8_t* src_row = src.ptr<uint8_t>(n, 0, static_cast<size_t>(ih), static_cast<size_t>(iw0));
                int32_t* dst_row = dst.ptr<int32_t>(n, 0, oh, ow);
                const size_t valid = ow_end - ow_start + 1;
                size_t full = (valid / brgemm_1x1_m_blk_) * brgemm_1x1_m_blk_;
                size_t m = 0;
                for (; m < full; m += brgemm_1x1_m_blk_) {
                    const uint8_t* src_ptr = src_row + m * lda;
                    int32_t* dst_ptr = dst_row + m * ldc;
                    brgemm_kernel->execute(src_ptr, packed_wei_1x1_ptr, dst_ptr);
                    if (bias_ptr || use_comp) {
                        for (size_t bi = 0; bi < brgemm_1x1_m_blk_; ++bi) {
                            int32_t* dst_block = dst_ptr + bi * ldc;
                            for (size_t oc = 0; oc < OC; ++oc) {
                                const int32_t bias_val = bias_ptr ? bias_ptr[oc] : 0;
                                const int32_t comp_val = comp_ptr ? comp_ptr[oc] : 0;
                                dst_block[oc] += bias_val + comp_val;
                            }
                        }
                    }
                }
                for (; m < valid; ++m) {
                    const uint8_t* src_ptr =
                        src.ptr<uint8_t>(n, 0, static_cast<size_t>(ih), static_cast<size_t>(iw0 + m * stride_w));
                    for (size_t oc = 0; oc < OC; ++oc) {
                        const int8_t* wei_ptr = wei.ptr<int8_t>(oc, 0, 0, 0);
                        int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow + m);
                        ker(src_ptr, wei_ptr, dst_ptr, IC, 0);
                        if (bias_ptr) {
                            *dst_ptr += bias_ptr[oc];
                        }
                    }
                }
                ow = ow_start + valid;
            }

            for (; ow < OW; ++ow) {
                int32_t* dst_ptr = dst.ptr<int32_t>(n, 0, oh, ow);
                if (bias_ptr) {
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dst_ptr[oc] = bias_ptr[oc];
                    }
                } else {
                    std::fill(dst_ptr, dst_ptr + OC, 0);
                }
            }
        }
    }

    return true;
}

void BrgemmInt8ConvExecutor::execute_impl(PlainTensor& dst, const MemoryArgs& memory) {
    PlainTensor src(memory.at(ARG_SRC));
    PlainTensor wei(memory.at(ARG_WEI));

    const bool src_signed = src.get_precision() == ov::element::i8;
    const auto ker = src_signed ? kernel_s8_->ker() : kernel_u8_->ker();
    const auto ker_block4_u8 = kernel_block4_u8_->ker();
    const auto ker_block4_s8 = kernel_block4_s8_->ker();
    const auto ker_block4_dot = kernel_block4_dot_s8_->ker();
    const auto ker_block4_udot = kernel_block4_udot_->ker();
    const auto ker_block8_dot = kernel_block8_dot_s8_->ker();
    const auto ker_block8_udot = kernel_block8_udot_->ker();
    const auto ker_block8_dot_packed = kernel_block8_dot_packed_s8_->ker();
    const auto ker_block8_udot_packed = kernel_block8_udot_packed_->ker();
    const auto ker_block2x8_dot_packed_strided = kernel_block2x8_dot_packed_s8_strided_->ker();
    const auto ker_block2x8_dot_packed_strided_interleaved =
        kernel_block2x8_dot_packed_s8_strided_interleaved_->ker();
    const auto ker_block2x16_dot_packed_strided_interleaved =
        kernel_block2x16_dot_packed_s8_strided_interleaved_->ker();
    const auto ker_block2x32_dot_packed_strided_interleaved =
        kernel_block2x32_dot_packed_s8_strided_interleaved_->ker();
    const auto ker_block4x16_dot_packed_strided_interleaved =
        kernel_block4x16_dot_packed_s8_strided_interleaved_->ker();
    const auto ker_block4x16_dot_packed_lhs_strided_interleaved =
        kernel_block4x16_dot_packed_lhs_s8_strided_interleaved_->ker();
    const auto ker_block2x8_udot_packed_strided = kernel_block2x8_udot_packed_strided_->ker();
    const auto ker_block2x8_udot_packed_strided_interleaved =
        kernel_block2x8_udot_packed_strided_interleaved_->ker();
    const auto ker_block2x16_udot_packed_strided_interleaved =
        kernel_block2x16_udot_packed_strided_interleaved_->ker();
    const auto ker_block2x32_udot_packed_strided_interleaved =
        kernel_block2x32_udot_packed_strided_interleaved_->ker();
    const auto ker_block4x16_udot_packed_strided_interleaved =
        kernel_block4x16_udot_packed_strided_interleaved_->ker();
    const auto ker_block4x16_udot_packed_lhs_strided_interleaved =
        kernel_block4x16_udot_packed_lhs_strided_interleaved_->ker();
    const auto ker_block4x4_dot = kernel_block4x4_dot_s8_->ker();
    const auto ker_block4x4_mmla_packed_s8 = kernel_block4x4_mmla_packed_s8_->ker();
    const auto ker_block4x8_mmla_packed_s8 = kernel_block4x8_mmla_packed_s8_->ker();
    const auto ker_block4x16_mmla_packed_s8 = kernel_block4x16_mmla_packed_s8_->ker();
    const auto ker_block4x4_mmla_packed_s8_interleaved = kernel_block4x4_mmla_packed_s8_interleaved_->ker();
    const auto ker_block4x8_mmla_packed_s8_interleaved = kernel_block4x8_mmla_packed_s8_interleaved_->ker();
    const auto ker_block8x8_mmla_packed_s8 = kernel_block8x8_mmla_packed_s8_->ker();
    const auto ker_block8x12_mmla_packed_s8 = kernel_block8x12_mmla_packed_s8_->ker();
    const auto ker_block4x16_mmla_packed_s8_interleaved = kernel_block4x16_mmla_packed_s8_interleaved_->ker();
    const auto ker_block4x4_udot = kernel_block4x4_udot_->ker();
    const auto ker_block4x4_dot_packed = kernel_block4x4_dot_packed_s8_->ker();
    const auto ker_block4x4_mmla_packed_u8 = kernel_block4x4_mmla_packed_u8_->ker();
    const auto ker_block4x8_mmla_packed_u8 = kernel_block4x8_mmla_packed_u8_->ker();
    const auto ker_block4x16_mmla_packed_u8 = kernel_block4x16_mmla_packed_u8_->ker();
    const auto ker_block4x4_mmla_packed_u8_interleaved = kernel_block4x4_mmla_packed_u8_interleaved_->ker();
    const auto ker_block4x8_mmla_packed_u8_interleaved = kernel_block4x8_mmla_packed_u8_interleaved_->ker();
    const auto ker_block8x8_mmla_packed_u8 = kernel_block8x8_mmla_packed_u8_->ker();
    const auto ker_block8x12_mmla_packed_u8 = kernel_block8x12_mmla_packed_u8_->ker();
    const auto ker_block4x16_mmla_packed_u8_interleaved = kernel_block4x16_mmla_packed_u8_interleaved_->ker();
    const auto ker_block4x4_udot_packed = kernel_block4x4_udot_packed_->ker();
    const bool prefer_sve_int8 = ov::intel_cpu::isSVEInt8Isa(runtime_isa_);
    const bool use_dot_s8 = src_signed && has_dotprod_ && !prefer_sve_int8;
    const bool use_dot_u8 = !src_signed && has_dotprod_ && !prefer_sve_int8;
    const bool use_mmla_s8 = src_signed && has_i8mm_ && !prefer_sve_int8;
    const bool use_mmla_u8 = !src_signed && has_i8mm_ && !prefer_sve_int8;

    const auto& src_dims = src.shape();
    const auto& dst_dims = dst.shape();
    const size_t N = src_dims[0];
    const size_t IC = src_dims[1];
    const size_t IH = src_dims[2];
    const size_t IW = src_dims[3];
    const size_t OH = dst_dims[2];
    const size_t OW = dst_dims[3];
    const size_t OC = dst_dims[1];
    const auto& wei_dims = wei.shape();
    const size_t wei_rank = wei_dims.size();
    const size_t KH = wei_dims[wei_rank - 2];
    const size_t KW = wei_dims[wei_rank - 1];
    const bool small_ow_1x1 = (KH == 1) && (KW == 1) && (OW < 16);
    const bool use_block4x4_dot = use_dot_s8 && (KH == 1) && (KW == 1);
    const bool use_block4x4_udot = use_dot_u8 && (KH == 1) && (KW == 1);
    const bool use_block4x16_mmla =
        (use_mmla_s8 || use_mmla_u8) && !small_ow_1x1 && (KH == 1) && (KW == 1) &&
        (IC % 8 == 0) &&
        !packed_wei_1x1_mmla16_.empty();
    const bool use_block4x8_mmla =
        (use_mmla_s8 || use_mmla_u8) && !small_ow_1x1 && (KH == 1) && (KW == 1) &&
        (IC % 8 == 0) &&
        !packed_wei_1x1_mmla8_.empty();
    const bool use_block4x4_mmla =
        (use_mmla_s8 || use_mmla_u8) && !small_ow_1x1 && (KH == 1) && (KW == 1) &&
        (IC % 8 == 0) &&
        !packed_wei_1x1_mmla4_.empty();
    const bool use_block8x8_mmla =
        (use_mmla_s8 || use_mmla_u8) && !small_ow_1x1 && (KH == 1) && (KW == 1) &&
        (IC % 8 == 0) &&
        (OC % 8 == 0) && !packed_wei_1x1_mmla8_.empty();
    const bool use_block8x12_mmla =
        (use_mmla_s8 || use_mmla_u8) && !small_ow_1x1 && (KH == 1) && (KW == 1) &&
        (IC % 8 == 0) &&
        !packed_wei_1x1_mmla12_.empty();
    const bool use_packed4_1x1 = has_dotprod_ && !packed_wei_1x1_dot4_.empty();
    const bool use_packed8_1x1 = has_dotprod_ && !packed_wei_1x1_dot8_.empty();
    const bool use_block2x8_dot = use_dot_s8 && use_packed8_1x1 && (OC % 8 == 0);
    const bool use_block2x8_udot = use_dot_u8 && use_packed8_1x1 && (OC % 8 == 0);
    const bool enable_block4x16_interleaved = true;
    const bool enable_block2x16_interleaved = true;
    const bool enable_block2x16_udot_interleaved = false;  // Slower than block2x8 in current microbench.
    const bool enable_block2x32_interleaved = true;
    const bool enable_block2x32_udot_interleaved = true;  // Enabled for perf parity check.
    const bool use_block2x32_dot_interleaved =
        enable_block2x32_interleaved && use_dot_s8 && (OC % 32 == 0) && (IC % 4 == 0) &&
        !packed_wei_1x1_dot32_interleaved_.empty();
    const bool use_block2x32_udot_interleaved =
        enable_block2x32_udot_interleaved && use_dot_u8 && (OC % 32 == 0) &&
        (IC % 4 == 0) &&
        !packed_wei_1x1_dot32_interleaved_.empty();
    const bool use_block4x16_dot_interleaved =
        enable_block4x16_interleaved && use_dot_s8 && (OC % 16 == 0) && (IC % 4 == 0) && (OW >= 4) &&
        !packed_wei_1x1_dot16_interleaved_.empty();
    const bool use_block4x16_udot_interleaved =
        enable_block4x16_interleaved && use_dot_u8 && (OC % 16 == 0) && (IC % 4 == 0) && (OW >= 4) &&
        !packed_wei_1x1_dot16_interleaved_.empty() && !use_block2x32_udot_interleaved;
    const bool use_block4x16_dot_interleaved_lhs = use_block4x16_dot_interleaved && (IC >= 32);
    const bool use_block4x16_udot_interleaved_lhs = use_block4x16_udot_interleaved && (IC >= 32);
    const bool use_block2x16_dot_interleaved =
        enable_block2x16_interleaved && use_dot_s8 && (OC % 16 == 0) && (IC % 4 == 0) &&
        !packed_wei_1x1_dot16_interleaved_.empty();
    const bool use_block2x16_udot_interleaved =
        enable_block2x16_udot_interleaved && use_dot_u8 && (OC % 16 == 0) && (IC % 4 == 0) &&
        !packed_wei_1x1_dot16_interleaved_.empty();
    const bool enable_block2x8_interleaved = false;  // Slower than baseline in current microbench.
    const bool use_block2x8_dot_interleaved =
        enable_block2x8_interleaved && use_block2x8_dot && (IC % 4 == 0) && !packed_wei_1x1_dot8_interleaved_.empty();
    const bool use_block2x8_udot_interleaved =
        enable_block2x8_interleaved && use_block2x8_udot && (IC % 4 == 0) && !packed_wei_1x1_dot8_interleaved_.empty();
    const size_t conv_stride_h = attrs_.stride[0];
    const size_t conv_stride_w = attrs_.stride[1];
    const ptrdiff_t pad_t = attrs_.paddingL[0];
    const ptrdiff_t pad_l = attrs_.paddingL[1];
    const ptrdiff_t pad_r = attrs_.paddingR[1];
    const ptrdiff_t pad_b = attrs_.paddingR[0];
    const bool prefer_block2x8_1x1 =
        small_ow_1x1 &&
        (use_block2x8_dot || use_block2x8_udot || use_block2x16_dot_interleaved || use_block2x16_udot_interleaved ||
         use_block2x32_dot_interleaved || use_block2x32_udot_interleaved) &&
        (conv_stride_w == 1) && (pad_l == 0) && (pad_r == 0);
    const bool nhwc_contiguous =
        (src.stride(3) == 1) && (dst.stride(3) == 1) && (src.stride(2) == IC) && (dst.stride(2) == OC) &&
        (src.stride(1) == IW * IC) && (dst.stride(1) == OW * OC);
    const bool use_gemm_mmla_small_ow =
        small_ow_1x1 && nhwc_contiguous && (conv_stride_w == 1) && (conv_stride_h == 1) && (pad_l == 0) &&
        (pad_r == 0) && (pad_t == 0) && (pad_b == 0) && (IH == OH) && (IW == OW) &&
        (use_mmla_s8 || use_mmla_u8) && (IC % 8 == 0);
    if (!logged_impl_path_) {
        const char* path = "scalar_reference";
        if (KH == 1 && KW == 1) {
            if (use_gemm_mmla_small_ow) {
                path = "conv1x1_mmla_small_ow";
            } else if (use_block2x32_udot_interleaved) {
                path = "conv1x1_dot_2x32_u8";
            } else if (use_block2x32_dot_interleaved) {
                path = "conv1x1_dot_2x32_s8";
            } else if (use_block4x16_udot_interleaved_lhs) {
                path = "conv1x1_dot_4x16_u8_lhs";
            } else if (use_block4x16_dot_interleaved_lhs) {
                path = "conv1x1_dot_4x16_s8_lhs";
            } else if (use_block4x16_udot_interleaved) {
                path = "conv1x1_dot_4x16_u8";
            } else if (use_block4x16_dot_interleaved) {
                path = "conv1x1_dot_4x16_s8";
            } else if (use_block2x16_udot_interleaved) {
                path = "conv1x1_dot_2x16_u8";
            } else if (use_block2x16_dot_interleaved) {
                path = "conv1x1_dot_2x16_s8";
            } else if (use_block2x8_udot_interleaved) {
                path = "conv1x1_dot_2x8_u8_interleaved";
            } else if (use_block2x8_dot_interleaved) {
                path = "conv1x1_dot_2x8_s8_interleaved";
            } else if (use_block2x8_udot) {
                path = "conv1x1_dot_2x8_u8";
            } else if (use_block2x8_dot) {
                path = "conv1x1_dot_2x8_s8";
            } else if (use_block8x12_mmla) {
                path = "conv1x1_mmla_8x12";
            } else if (use_block8x8_mmla) {
                path = "conv1x1_mmla_8x8";
            } else if (use_block4x16_mmla) {
                path = "conv1x1_mmla_4x16";
            } else if (use_block4x8_mmla) {
                path = "conv1x1_mmla_4x8";
            } else if (use_block4x4_mmla) {
                path = "conv1x1_mmla_4x4";
            } else if (use_block4x4_udot) {
                path = "conv1x1_dot_4x4_u8";
            } else if (use_block4x4_dot) {
                path = "conv1x1_dot_4x4_s8";
            }
        } else if (use_mmla_s8 || use_mmla_u8) {
            path = "conv_kxk_mmla";
        } else if (use_dot_s8 || use_dot_u8) {
            path = "conv_kxk_dotprod";
        }
        jit_int8_conv_debug("selected_path=", path,
                            " src=", src_signed ? "s8" : "u8",
                            " KH=", KH,
                            " KW=", KW,
                            " IC=", IC,
                            " OC=", OC,
                            " OW=", OW);
        logged_impl_path_ = true;
    }

    const auto bias_prec = (bias_prec_ == ov::element::dynamic && attrs_.withBias)
                               ? memory.at(ARG_BIAS)->getDescPtr()->getPrecision()
                               : bias_prec_;
    const bool bias_i32 = bias_prec == ov::element::i32;
    const int32_t* bias_ptr = nullptr;
    PlainTensor bias;
    if (attrs_.withBias && bias_i32) {
        bias.reset(memory.at(ARG_BIAS));
        bias_ptr = bias.ptr<int32_t>(0);
    }
    const int32_t* bias_ptr_base = bias_ptr;

    std::vector<uint8_t> packed_src_u8;
    std::vector<int8_t> packed_src_s8;
    std::vector<uint8_t> packed_lhs4x16_u8;
    std::vector<int8_t> packed_lhs4x16_s8;
    const size_t packed_pair_stride = IC * 2;
    const bool use_interleaved4x_mmla =
        (use_block4x4_mmla || use_block4x8_mmla || use_block4x16_mmla) && (conv_stride_w == 1) && (pad_l == 0) &&
        (pad_r == 0) && (IC % 8 == 0);
    if (use_block8x8_mmla || use_interleaved4x_mmla) {
        const size_t pack_pairs = use_block8x8_mmla ? 4 : 2;
        const size_t pack_bytes = packed_pair_stride * pack_pairs;
        if (src_signed) {
            packed_src_s8.resize(pack_bytes);
        } else {
            packed_src_u8.resize(pack_bytes);
        }
    }
    const bool use_block4x16_lhs_pack =
        (use_block4x16_dot_interleaved_lhs || use_block4x16_udot_interleaved_lhs) && (conv_stride_w == 1) &&
        (pad_l == 0) && (pad_r == 0);
    const size_t packed_lhs4x16_row_stride = IC * 4;
    const size_t packed_lhs4x16_block_stride = packed_lhs4x16_row_stride * 4;
    if (use_block4x16_lhs_pack) {
        const size_t pack_blocks = (OW + 3) / 4;
        const size_t pack_bytes = packed_lhs4x16_block_stride * pack_blocks;
        if (src_signed) {
            packed_lhs4x16_s8.resize(pack_bytes);
        } else {
            packed_lhs4x16_u8.resize(pack_bytes);
        }
    }

    if (KH == 1 && KW == 1) {
        const bool use_bias_comp_1x1 = !src_signed && !bias_comp_1x1_.empty();
        const int32_t* bias_ptr = use_bias_comp_1x1 ? bias_comp_1x1_.data() : bias_ptr_base;
        const bool add_comp_1x1 = !src_signed && !wei_comp_1x1_.empty() && !use_bias_comp_1x1;
        if (use_gemm_mmla_small_ow) {
            const bool use_mmla16 = !packed_wei_1x1_mmla16_.empty();
            const bool use_mmla8 = !packed_wei_1x1_mmla8_.empty();
            const bool use_mmla4 = !packed_wei_1x1_mmla4_.empty();
            const size_t M = OH * OW;
            const size_t src_row_stride = IC;
            const size_t dst_row_stride = OC;
            const size_t dst_row_stride_bytes = dst_row_stride * sizeof(int32_t);
            for (size_t n = 0; n < N; ++n) {
                const uint8_t* src_base = src.ptr<uint8_t>(n, 0, 0, 0);
                int32_t* dst_base = dst.ptr<int32_t>(n, 0, 0, 0);
                size_t m = 0;
                for (; m + 4 <= M; m += 4) {
                    const uint8_t* src_ptr0 = src_base + (m + 0) * src_row_stride;
                    const uint8_t* src_ptr1 = src_base + (m + 1) * src_row_stride;
                    const uint8_t* src_ptr2 = src_base + (m + 2) * src_row_stride;
                    const uint8_t* src_ptr3 = src_base + (m + 3) * src_row_stride;
                    const int8_t* src_ptrs_s8[4] = {
                        reinterpret_cast<const int8_t*>(src_ptr0),
                        reinterpret_cast<const int8_t*>(src_ptr1),
                        reinterpret_cast<const int8_t*>(src_ptr2),
                        reinterpret_cast<const int8_t*>(src_ptr3),
                    };
                    const uint8_t* src_ptrs_u8[4] = {src_ptr0, src_ptr1, src_ptr2, src_ptr3};
                    size_t oc = 0;
                    for (; use_mmla16 && oc + 16 <= OC; oc += 16) {
                        const int8_t* wei_ptr =
                            packed_wei_1x1_mmla16_.data() + (oc / 16) * packed_wei_1x1_mmla16_stride_;
                        int32_t* dst_ptr = dst_base + m * dst_row_stride + oc;
                        if (use_mmla_s8) {
                            ker_block4x16_mmla_packed_s8(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, dst_row_stride_bytes, 0);
                        } else {
                            ker_block4x16_mmla_packed_u8(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, dst_row_stride_bytes, 0);
                            if (add_comp_1x1) {
                                for (size_t r = 0; r < 4; ++r) {
                                    int32_t* dst_row = dst_ptr + r * dst_row_stride;
                                    for (size_t c = 0; c < 16; ++c) {
                                        dst_row[c] += wei_comp_1x1_[oc + c];
                                    }
                                }
                            }
                        }
                        if (bias_ptr) {
                            int32_t* dst_row0 = dst_ptr;
                            int32_t* dst_row1 = dst_row0 + dst_row_stride;
                            int32_t* dst_row2 = dst_row1 + dst_row_stride;
                            int32_t* dst_row3 = dst_row2 + dst_row_stride;
                            for (size_t c = 0; c < 16; ++c) {
                                const int32_t b = bias_ptr[oc + c];
                                dst_row0[c] += b;
                                dst_row1[c] += b;
                                dst_row2[c] += b;
                                dst_row3[c] += b;
                            }
                        }
                    }
                    for (; use_mmla8 && oc + 8 <= OC; oc += 8) {
                        const int8_t* wei_ptr =
                            packed_wei_1x1_mmla8_.data() + (oc / 8) * packed_wei_1x1_mmla8_stride_;
                        int32_t* dst_ptr = dst_base + m * dst_row_stride + oc;
                        if (use_mmla_s8) {
                            ker_block4x8_mmla_packed_s8(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, dst_row_stride_bytes, 0);
                        } else {
                            ker_block4x8_mmla_packed_u8(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, dst_row_stride_bytes, 0);
                            if (add_comp_1x1) {
                                for (size_t r = 0; r < 4; ++r) {
                                    int32_t* dst_row = dst_ptr + r * dst_row_stride;
                                    for (size_t c = 0; c < 8; ++c) {
                                        dst_row[c] += wei_comp_1x1_[oc + c];
                                    }
                                }
                            }
                        }
                        if (bias_ptr) {
                            int32_t* dst_row0 = dst_ptr;
                            int32_t* dst_row1 = dst_row0 + dst_row_stride;
                            int32_t* dst_row2 = dst_row1 + dst_row_stride;
                            int32_t* dst_row3 = dst_row2 + dst_row_stride;
                            for (size_t c = 0; c < 8; ++c) {
                                const int32_t b = bias_ptr[oc + c];
                                dst_row0[c] += b;
                                dst_row1[c] += b;
                                dst_row2[c] += b;
                                dst_row3[c] += b;
                            }
                        }
                    }
                    for (; use_mmla4 && oc + 4 <= OC; oc += 4) {
                        const int8_t* wei_ptr =
                            packed_wei_1x1_mmla4_.data() + (oc / 4) * packed_wei_1x1_mmla4_stride_;
                        int32_t* dst_ptr = dst_base + m * dst_row_stride + oc;
                        if (use_mmla_s8) {
                            ker_block4x4_mmla_packed_s8(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, dst_row_stride_bytes, 0);
                        } else {
                            ker_block4x4_mmla_packed_u8(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, dst_row_stride_bytes, 0);
                            if (add_comp_1x1) {
                                for (size_t r = 0; r < 4; ++r) {
                                    int32_t* dst_row = dst_ptr + r * dst_row_stride;
                                    for (size_t c = 0; c < 4; ++c) {
                                        dst_row[c] += wei_comp_1x1_[oc + c];
                                    }
                                }
                            }
                        }
                        if (bias_ptr) {
                            int32_t* dst_row0 = dst_ptr;
                            int32_t* dst_row1 = dst_row0 + dst_row_stride;
                            int32_t* dst_row2 = dst_row1 + dst_row_stride;
                            int32_t* dst_row3 = dst_row2 + dst_row_stride;
                            for (size_t c = 0; c < 4; ++c) {
                                const int32_t b = bias_ptr[oc + c];
                                dst_row0[c] += b;
                                dst_row1[c] += b;
                                dst_row2[c] += b;
                                dst_row3[c] += b;
                            }
                        }
                    }
                    for (; oc < OC; ++oc) {
                        const int8_t* wei_ptr = wei.ptr<int8_t>(oc, 0, 0, 0);
                        int32_t* dst_ptr = dst_base + m * dst_row_stride + oc;
                        if (use_mmla_s8) {
                            ker(reinterpret_cast<const uint8_t*>(src_ptrs_s8[0]), wei_ptr, dst_ptr, IC, 0);
                            ker(reinterpret_cast<const uint8_t*>(src_ptrs_s8[1]),
                                wei_ptr,
                                dst_ptr + dst_row_stride,
                                IC,
                                0);
                            ker(reinterpret_cast<const uint8_t*>(src_ptrs_s8[2]),
                                wei_ptr,
                                dst_ptr + 2 * dst_row_stride,
                                IC,
                                0);
                            ker(reinterpret_cast<const uint8_t*>(src_ptrs_s8[3]),
                                wei_ptr,
                                dst_ptr + 3 * dst_row_stride,
                                IC,
                                0);
                        } else {
                            ker(src_ptrs_u8[0], wei_ptr, dst_ptr, IC, 0);
                            ker(src_ptrs_u8[1], wei_ptr, dst_ptr + dst_row_stride, IC, 0);
                            ker(src_ptrs_u8[2], wei_ptr, dst_ptr + 2 * dst_row_stride, IC, 0);
                            ker(src_ptrs_u8[3], wei_ptr, dst_ptr + 3 * dst_row_stride, IC, 0);
                        }
                        if (add_comp_1x1) {
                            dst_ptr[0] += wei_comp_1x1_[oc];
                            dst_ptr[dst_row_stride] += wei_comp_1x1_[oc];
                            dst_ptr[2 * dst_row_stride] += wei_comp_1x1_[oc];
                            dst_ptr[3 * dst_row_stride] += wei_comp_1x1_[oc];
                        }
                        if (bias_ptr) {
                            const int32_t b = bias_ptr[oc];
                            dst_ptr[0] += b;
                            dst_ptr[dst_row_stride] += b;
                            dst_ptr[2 * dst_row_stride] += b;
                            dst_ptr[3 * dst_row_stride] += b;
                        }
                    }
                }
                for (; m < M; ++m) {
                    const uint8_t* src_ptr = src_base + m * src_row_stride;
                    int32_t* dst_ptr = dst_base + m * dst_row_stride;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        const int8_t* wei_ptr = wei.ptr<int8_t>(oc, 0, 0, 0);
                        ker(src_ptr, wei_ptr, dst_ptr + oc, IC, 0);
                        if (add_comp_1x1) {
                            dst_ptr[oc] += wei_comp_1x1_[oc];
                        }
                        if (bias_ptr) {
                            dst_ptr[oc] += bias_ptr[oc];
                        }
                    }
                }
            }
            return;
        }
        for (size_t n = 0; n < N; n++) {
            for (size_t oh = 0; oh < OH; oh++) {
                const ptrdiff_t ih = static_cast<ptrdiff_t>(oh * conv_stride_h) - pad_t;
                size_t ow = 0;
                const bool use_ow8_mmla =
                    (use_block8x8_mmla || use_block8x12_mmla) && (conv_stride_w == 1) && (pad_l == 0) &&
                    (pad_r == 0);
                if (use_ow8_mmla && ih >= 0 && ih < static_cast<ptrdiff_t>(IH) && OW >= 8) {
                    const uint8_t* src_row = src.ptr<uint8_t>(n, 0, static_cast<size_t>(ih), 0);
                    const bool prefer_block8x8_for_oc24 = use_block8x8_mmla && (KH == 1) && (KW == 1) && (OC == 24);
                    for (; ow + 8 <= OW; ow += 8) {
                        const uint8_t* src_base = src_row + ow * IC;
                        const bool use_12_block_strategy =
                            use_block8x12_mmla && !prefer_block8x8_for_oc24 &&
                            ((OC % 12) == 0 || (OC % 12) == 4 || (OC % 12) == 8);
                        if (src_signed) {
                            int8_t* pair_ptrs[4] = {packed_src_s8.data(),
                                                    packed_src_s8.data() + packed_pair_stride,
                                                    packed_src_s8.data() + 2 * packed_pair_stride,
                                                    packed_src_s8.data() + 3 * packed_pair_stride};
                            for (size_t p = 0; p < 4; ++p) {
                                const int8_t* row0 = reinterpret_cast<const int8_t*>(src_base + (2 * p) * IC);
                                const int8_t* row1 = reinterpret_cast<const int8_t*>(src_base + (2 * p + 1) * IC);
                                int8_t* dst_pair = pair_ptrs[p];
                                for (size_t icb = 0; icb < IC; icb += 8) {
                                    std::memcpy(dst_pair, row0 + icb, 8);
                                    std::memcpy(dst_pair + 8, row1 + icb, 8);
                                    dst_pair += 16;
                                }
                            }
                            const int8_t* src_ptrs_s8[4] = {pair_ptrs[0], pair_ptrs[1], pair_ptrs[2], pair_ptrs[3]};
                            size_t oc = 0;
                            if (use_12_block_strategy) {
                                const size_t n12_main = (OC / 12) * 12;
                                const size_t rem12 = OC - n12_main;
                                for (; oc + 12 <= n12_main; oc += 12) {
                                    const int8_t* wei_ptr =
                                        packed_wei_1x1_mmla12_.data() + (oc / 12) * packed_wei_1x1_mmla12_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    ker_block8x12_mmla_packed_s8(src_ptrs_s8,
                                                                 wei_ptr,
                                                                 dst_ptr,
                                                                 IC,
                                                                 0,
                                                                 OC * sizeof(int32_t),
                                                                 0);
                                    if (bias_ptr) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 12; ++c) {
                                                dst_row[c] += bias_ptr[oc + c];
                                            }
                                        }
                                    }
                                }
                                if (rem12 >= 8) {
                                    const int8_t* wei_ptr =
                                        packed_wei_1x1_mmla8_.data() + (oc / 8) * packed_wei_1x1_mmla8_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    ker_block8x8_mmla_packed_s8(src_ptrs_s8,
                                                                wei_ptr,
                                                                dst_ptr,
                                                                IC,
                                                                0,
                                                                OC * sizeof(int32_t),
                                                                0);
                                    if (bias_ptr) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 8; ++c) {
                                                dst_row[c] += bias_ptr[oc + c];
                                            }
                                        }
                                    }
                                    oc += 8;
                                }
                            }
                            for (; oc + 8 <= OC; oc += 8) {
                                const int8_t* wei_ptr =
                                    packed_wei_1x1_mmla8_.data() + (oc / 8) * packed_wei_1x1_mmla8_stride_;
                                int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                ker_block8x8_mmla_packed_s8(src_ptrs_s8,
                                                            wei_ptr,
                                                            dst_ptr,
                                                            IC,
                                                            0,
                                                            OC * sizeof(int32_t),
                                                            0);
                                if (bias_ptr) {
                                    for (size_t r = 0; r < 8; ++r) {
                                        int32_t* dst_row = dst_ptr + r * OC;
                                        for (size_t c = 0; c < 8; ++c) {
                                            dst_row[c] += bias_ptr[oc + c];
                                        }
                                    }
                                }
                            }
                        } else {
                            uint8_t* pair_ptrs[4] = {packed_src_u8.data(),
                                                     packed_src_u8.data() + packed_pair_stride,
                                                     packed_src_u8.data() + 2 * packed_pair_stride,
                                                     packed_src_u8.data() + 3 * packed_pair_stride};
                            for (size_t p = 0; p < 4; ++p) {
                                const uint8_t* row0 = src_base + (2 * p) * IC;
                                const uint8_t* row1 = src_base + (2 * p + 1) * IC;
                                uint8_t* dst_pair = pair_ptrs[p];
                                for (size_t icb = 0; icb < IC; icb += 8) {
                                    std::memcpy(dst_pair, row0 + icb, 8);
                                    std::memcpy(dst_pair + 8, row1 + icb, 8);
                                    dst_pair += 16;
                                }
                            }
                            const uint8_t* src_ptrs_u8[4] = {pair_ptrs[0], pair_ptrs[1], pair_ptrs[2], pair_ptrs[3]};
                            size_t oc = 0;
                            if (use_12_block_strategy) {
                                const size_t n12_main = (OC / 12) * 12;
                                const size_t rem12 = OC - n12_main;
                                for (; oc + 12 <= n12_main; oc += 12) {
                                    const int8_t* wei_ptr =
                                        packed_wei_1x1_mmla12_.data() + (oc / 12) * packed_wei_1x1_mmla12_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    const int32_t* bias_block = use_bias_comp_1x1 ? bias_ptr + oc : nullptr;
                                    ker_block8x12_mmla_packed_u8(src_ptrs_u8,
                                                                 wei_ptr,
                                                                 dst_ptr,
                                                                 IC,
                                                                 bias_block,
                                                                 OC * sizeof(int32_t),
                                                                 0);
                                    if (add_comp_1x1) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 12; ++c) {
                                                dst_row[c] += wei_comp_1x1_[oc + c];
                                            }
                                        }
                                    }
                                    if (bias_ptr && !use_bias_comp_1x1) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 12; ++c) {
                                                dst_row[c] += bias_ptr[oc + c];
                                            }
                                        }
                                    }
                                }
                                if (rem12 >= 8) {
                                    const int8_t* wei_ptr =
                                        packed_wei_1x1_mmla8_.data() + (oc / 8) * packed_wei_1x1_mmla8_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    const int32_t* bias_block = use_bias_comp_1x1 ? bias_ptr + oc : nullptr;
                                    ker_block8x8_mmla_packed_u8(src_ptrs_u8,
                                                                wei_ptr,
                                                                dst_ptr,
                                                                IC,
                                                                bias_block,
                                                                OC * sizeof(int32_t),
                                                                0);
                                    if (add_comp_1x1) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 8; ++c) {
                                                dst_row[c] += wei_comp_1x1_[oc + c];
                                            }
                                        }
                                    }
                                    if (bias_ptr && !use_bias_comp_1x1) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 8; ++c) {
                                                dst_row[c] += bias_ptr[oc + c];
                                            }
                                        }
                                    }
                                    oc += 8;
                                }
                            }
                            for (; oc + 8 <= OC; oc += 8) {
                                const int8_t* wei_ptr =
                                    packed_wei_1x1_mmla8_.data() + (oc / 8) * packed_wei_1x1_mmla8_stride_;
                                int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                const int32_t* bias_block = use_bias_comp_1x1 ? bias_ptr + oc : nullptr;
                                ker_block8x8_mmla_packed_u8(src_ptrs_u8,
                                                            wei_ptr,
                                                            dst_ptr,
                                                            IC,
                                                            bias_block,
                                                            OC * sizeof(int32_t),
                                                            0);
                                if (add_comp_1x1) {
                                    for (size_t r = 0; r < 8; ++r) {
                                        int32_t* dst_row = dst_ptr + r * OC;
                                        for (size_t c = 0; c < 8; ++c) {
                                            dst_row[c] += wei_comp_1x1_[oc + c];
                                        }
                                    }
                                }
                                if (bias_ptr && !use_bias_comp_1x1) {
                                    for (size_t r = 0; r < 8; ++r) {
                                        int32_t* dst_row = dst_ptr + r * OC;
                                        for (size_t c = 0; c < 8; ++c) {
                                            dst_row[c] += bias_ptr[oc + c];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (!prefer_block2x8_1x1 &&
                    (use_block4x4_dot || use_block4x4_udot || use_block4x4_mmla || use_block4x8_mmla ||
                     use_block4x16_mmla) &&
                    OW >= 4) {
                    for (; ow + 4 <= OW; ow += 4) {
                        const ptrdiff_t iw0 = static_cast<ptrdiff_t>(ow * conv_stride_w) - pad_l;
                        const ptrdiff_t iw1 = iw0 + static_cast<ptrdiff_t>(conv_stride_w);
                        const ptrdiff_t iw2 = iw1 + static_cast<ptrdiff_t>(conv_stride_w);
                        const ptrdiff_t iw3 = iw2 + static_cast<ptrdiff_t>(conv_stride_w);

                        const bool in0 = ih >= 0 && ih < static_cast<ptrdiff_t>(IH) && iw0 >= 0 &&
                                         iw0 < static_cast<ptrdiff_t>(IW);
                        const bool in1 = ih >= 0 && ih < static_cast<ptrdiff_t>(IH) && iw1 >= 0 &&
                                         iw1 < static_cast<ptrdiff_t>(IW);
                        const bool in2 = ih >= 0 && ih < static_cast<ptrdiff_t>(IH) && iw2 >= 0 &&
                                         iw2 < static_cast<ptrdiff_t>(IW);
                        const bool in3 = ih >= 0 && ih < static_cast<ptrdiff_t>(IH) && iw3 >= 0 &&
                                         iw3 < static_cast<ptrdiff_t>(IW);

                        const uint8_t* src_ptrs_u8[4] = {
                            in0 ? src.ptr<uint8_t>(n, 0, static_cast<size_t>(ih), static_cast<size_t>(iw0)) : nullptr,
                            in1 ? src.ptr<uint8_t>(n, 0, static_cast<size_t>(ih), static_cast<size_t>(iw1)) : nullptr,
                            in2 ? src.ptr<uint8_t>(n, 0, static_cast<size_t>(ih), static_cast<size_t>(iw2)) : nullptr,
                            in3 ? src.ptr<uint8_t>(n, 0, static_cast<size_t>(ih), static_cast<size_t>(iw3)) : nullptr,
                        };
                        size_t oc = 0;
                        if (in0 && in1 && in2 && in3 && use_interleaved4x_mmla) {
                            const size_t ic_blocks = IC / 8;
                            if (src_signed) {
                                int8_t* packed0 = packed_src_s8.data();
                                int8_t* packed1 = packed_src_s8.data() + packed_pair_stride;
                                const int8_t* src0 = reinterpret_cast<const int8_t*>(src_ptrs_u8[0]);
                                const int8_t* src1 = reinterpret_cast<const int8_t*>(src_ptrs_u8[1]);
                                const int8_t* src2 = reinterpret_cast<const int8_t*>(src_ptrs_u8[2]);
                                const int8_t* src3 = reinterpret_cast<const int8_t*>(src_ptrs_u8[3]);
                                pack_pairs4x8(reinterpret_cast<const uint8_t*>(src0),
                                              reinterpret_cast<const uint8_t*>(src1),
                                              reinterpret_cast<const uint8_t*>(src2),
                                              reinterpret_cast<const uint8_t*>(src3),
                                              reinterpret_cast<uint8_t*>(packed0),
                                              reinterpret_cast<uint8_t*>(packed1),
                                              ic_blocks);
                                const int8_t* src_ptrs_s8[4] = {packed0, packed1, nullptr, nullptr};
                                for (; use_block4x16_mmla && oc + 16 <= OC; oc += 16) {
                                    const int8_t* wei_ptr = packed_wei_1x1_mmla16_.data() +
                                                            (oc / 16) * packed_wei_1x1_mmla16_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    ker_block4x16_mmla_packed_s8_interleaved(src_ptrs_s8,
                                                                             wei_ptr,
                                                                             dst_ptr,
                                                                             IC,
                                                                             0,
                                                                             OC * sizeof(int32_t),
                                                                             0);
                                    if (bias_ptr) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        for (size_t c = 0; c < 16; ++c) {
                                            const int32_t b = bias_ptr[oc + c];
                                            dst_row0[c] += b;
                                            dst_row1[c] += b;
                                            dst_row2[c] += b;
                                            dst_row3[c] += b;
                                        }
                                    }
                                }
                                for (; use_block4x8_mmla && oc + 8 <= OC; oc += 8) {
                                    const int8_t* wei_ptr =
                                        packed_wei_1x1_mmla8_.data() + (oc / 8) * packed_wei_1x1_mmla8_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    ker_block4x8_mmla_packed_s8_interleaved(src_ptrs_s8,
                                                                            wei_ptr,
                                                                            dst_ptr,
                                                                            IC,
                                                                            0,
                                                                            OC * sizeof(int32_t),
                                                                            0);
                                    if (bias_ptr) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        for (size_t c = 0; c < 8; ++c) {
                                            const int32_t b = bias_ptr[oc + c];
                                            dst_row0[c] += b;
                                            dst_row1[c] += b;
                                            dst_row2[c] += b;
                                            dst_row3[c] += b;
                                        }
                                    }
                                }
                                for (; use_block4x4_mmla && oc + 4 <= OC; oc += 4) {
                                    const int8_t* wei_ptr =
                                        packed_wei_1x1_mmla4_.data() + (oc / 4) * packed_wei_1x1_mmla4_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    ker_block4x4_mmla_packed_s8_interleaved(src_ptrs_s8,
                                                                            wei_ptr,
                                                                            dst_ptr,
                                                                            IC,
                                                                            0,
                                                                            OC * sizeof(int32_t),
                                                                            0);
                                    if (bias_ptr) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        for (size_t c = 0; c < 4; ++c) {
                                            const int32_t b = bias_ptr[oc + c];
                                            dst_row0[c] += b;
                                            dst_row1[c] += b;
                                            dst_row2[c] += b;
                                            dst_row3[c] += b;
                                        }
                                    }
                                }
                            } else {
                                uint8_t* packed0 = packed_src_u8.data();
                                uint8_t* packed1 = packed_src_u8.data() + packed_pair_stride;
                                pack_pairs4x8(src_ptrs_u8[0],
                                              src_ptrs_u8[1],
                                              src_ptrs_u8[2],
                                              src_ptrs_u8[3],
                                              packed0,
                                              packed1,
                                              ic_blocks);
                                const uint8_t* src_ptrs_u8_i[4] = {packed0, packed1, nullptr, nullptr};
                                for (; use_block4x16_mmla && oc + 16 <= OC; oc += 16) {
                                    const int8_t* wei_ptr = packed_wei_1x1_mmla16_.data() +
                                                            (oc / 16) * packed_wei_1x1_mmla16_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    ker_block4x16_mmla_packed_u8_interleaved(src_ptrs_u8_i,
                                                                             wei_ptr,
                                                                             dst_ptr,
                                                                             IC,
                                                                             0,
                                                                             OC * sizeof(int32_t),
                                                                             0);
                                    if (add_comp_1x1) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        for (size_t c = 0; c < 16; ++c) {
                                            const int32_t w = wei_comp_1x1_[oc + c];
                                            dst_row0[c] += w;
                                            dst_row1[c] += w;
                                            dst_row2[c] += w;
                                            dst_row3[c] += w;
                                        }
                                    }
                                    if (bias_ptr) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        for (size_t c = 0; c < 16; ++c) {
                                            const int32_t b = bias_ptr[oc + c];
                                            dst_row0[c] += b;
                                            dst_row1[c] += b;
                                            dst_row2[c] += b;
                                            dst_row3[c] += b;
                                        }
                                    }
                                }
                                for (; use_block4x8_mmla && oc + 8 <= OC; oc += 8) {
                                    const int8_t* wei_ptr =
                                        packed_wei_1x1_mmla8_.data() + (oc / 8) * packed_wei_1x1_mmla8_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    ker_block4x8_mmla_packed_u8_interleaved(src_ptrs_u8_i,
                                                                            wei_ptr,
                                                                            dst_ptr,
                                                                            IC,
                                                                            0,
                                                                            OC * sizeof(int32_t),
                                                                            0);
                                    if (add_comp_1x1) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        for (size_t c = 0; c < 8; ++c) {
                                            const int32_t w = wei_comp_1x1_[oc + c];
                                            dst_row0[c] += w;
                                            dst_row1[c] += w;
                                            dst_row2[c] += w;
                                            dst_row3[c] += w;
                                        }
                                    }
                                    if (bias_ptr) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        for (size_t c = 0; c < 8; ++c) {
                                            const int32_t b = bias_ptr[oc + c];
                                            dst_row0[c] += b;
                                            dst_row1[c] += b;
                                            dst_row2[c] += b;
                                            dst_row3[c] += b;
                                        }
                                    }
                                }
                                for (; use_block4x4_mmla && oc + 4 <= OC; oc += 4) {
                                    const int8_t* wei_ptr =
                                        packed_wei_1x1_mmla4_.data() + (oc / 4) * packed_wei_1x1_mmla4_stride_;
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    ker_block4x4_mmla_packed_u8_interleaved(src_ptrs_u8_i,
                                                                            wei_ptr,
                                                                            dst_ptr,
                                                                            IC,
                                                                            0,
                                                                            OC * sizeof(int32_t),
                                                                            0);
                                    if (add_comp_1x1) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        for (size_t c = 0; c < 4; ++c) {
                                            const int32_t w = wei_comp_1x1_[oc + c];
                                            dst_row0[c] += w;
                                            dst_row1[c] += w;
                                            dst_row2[c] += w;
                                            dst_row3[c] += w;
                                        }
                                    }
                                    if (bias_ptr) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        for (size_t c = 0; c < 4; ++c) {
                                            const int32_t b = bias_ptr[oc + c];
                                            dst_row0[c] += b;
                                            dst_row1[c] += b;
                                            dst_row2[c] += b;
                                            dst_row3[c] += b;
                                        }
                                    }
                                }
                            }
                        }
                        if (in0 && in1 && in2 && in3) {
                            for (; use_block4x16_mmla && oc + 16 <= OC; oc += 16) {
                                const int8_t* wei_ptr = packed_wei_1x1_mmla16_.data() +
                                                        (oc / 16) * packed_wei_1x1_mmla16_stride_;
                                int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                if (use_mmla_s8) {
                                    const int8_t* src_ptrs_s8[4] = {
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                    };
                                    ker_block4x16_mmla_packed_s8(src_ptrs_s8,
                                                                 wei_ptr,
                                                                 dst_ptr,
                                                                 IC,
                                                                 0,
                                                                 OC * sizeof(int32_t),
                                                                 0);
                                } else {
                                    ker_block4x16_mmla_packed_u8(src_ptrs_u8,
                                                                 wei_ptr,
                                                                 dst_ptr,
                                                                 IC,
                                                                 0,
                                                                 OC * sizeof(int32_t),
                                                                 0);
                                }
                                if (use_mmla_u8 && add_comp_1x1) {
                                    for (size_t r = 0; r < 4; ++r) {
                                        int32_t* dst_row = dst_ptr + r * OC;
                                        for (size_t c = 0; c < 16; ++c) {
                                            dst_row[c] += wei_comp_1x1_[oc + c];
                                        }
                                    }
                                }
                                if (bias_ptr) {
                                    int32_t* dst_row0 = dst_ptr;
                                    int32_t* dst_row1 = dst_row0 + OC;
                                    int32_t* dst_row2 = dst_row1 + OC;
                                    int32_t* dst_row3 = dst_row2 + OC;
                                    const int32_t b0 = bias_ptr[oc];
                                    const int32_t b1 = bias_ptr[oc + 1];
                                    const int32_t b2 = bias_ptr[oc + 2];
                                    const int32_t b3 = bias_ptr[oc + 3];
                                    const int32_t b4 = bias_ptr[oc + 4];
                                    const int32_t b5 = bias_ptr[oc + 5];
                                    const int32_t b6 = bias_ptr[oc + 6];
                                    const int32_t b7 = bias_ptr[oc + 7];
                                    const int32_t b8 = bias_ptr[oc + 8];
                                    const int32_t b9 = bias_ptr[oc + 9];
                                    const int32_t b10 = bias_ptr[oc + 10];
                                    const int32_t b11 = bias_ptr[oc + 11];
                                    const int32_t b12 = bias_ptr[oc + 12];
                                    const int32_t b13 = bias_ptr[oc + 13];
                                    const int32_t b14 = bias_ptr[oc + 14];
                                    const int32_t b15 = bias_ptr[oc + 15];
                                    dst_row0[0] += b0;
                                    dst_row0[1] += b1;
                                    dst_row0[2] += b2;
                                    dst_row0[3] += b3;
                                    dst_row0[4] += b4;
                                    dst_row0[5] += b5;
                                    dst_row0[6] += b6;
                                    dst_row0[7] += b7;
                                    dst_row0[8] += b8;
                                    dst_row0[9] += b9;
                                    dst_row0[10] += b10;
                                    dst_row0[11] += b11;
                                    dst_row0[12] += b12;
                                    dst_row0[13] += b13;
                                    dst_row0[14] += b14;
                                    dst_row0[15] += b15;
                                    dst_row1[0] += b0;
                                    dst_row1[1] += b1;
                                    dst_row1[2] += b2;
                                    dst_row1[3] += b3;
                                    dst_row1[4] += b4;
                                    dst_row1[5] += b5;
                                    dst_row1[6] += b6;
                                    dst_row1[7] += b7;
                                    dst_row1[8] += b8;
                                    dst_row1[9] += b9;
                                    dst_row1[10] += b10;
                                    dst_row1[11] += b11;
                                    dst_row1[12] += b12;
                                    dst_row1[13] += b13;
                                    dst_row1[14] += b14;
                                    dst_row1[15] += b15;
                                    dst_row2[0] += b0;
                                    dst_row2[1] += b1;
                                    dst_row2[2] += b2;
                                    dst_row2[3] += b3;
                                    dst_row2[4] += b4;
                                    dst_row2[5] += b5;
                                    dst_row2[6] += b6;
                                    dst_row2[7] += b7;
                                    dst_row2[8] += b8;
                                    dst_row2[9] += b9;
                                    dst_row2[10] += b10;
                                    dst_row2[11] += b11;
                                    dst_row2[12] += b12;
                                    dst_row2[13] += b13;
                                    dst_row2[14] += b14;
                                    dst_row2[15] += b15;
                                    dst_row3[0] += b0;
                                    dst_row3[1] += b1;
                                    dst_row3[2] += b2;
                                    dst_row3[3] += b3;
                                    dst_row3[4] += b4;
                                    dst_row3[5] += b5;
                                    dst_row3[6] += b6;
                                    dst_row3[7] += b7;
                                    dst_row3[8] += b8;
                                    dst_row3[9] += b9;
                                    dst_row3[10] += b10;
                                    dst_row3[11] += b11;
                                    dst_row3[12] += b12;
                                    dst_row3[13] += b13;
                                    dst_row3[14] += b14;
                                    dst_row3[15] += b15;
                                }
                            }
                            for (; use_block4x8_mmla && oc + 8 <= OC; oc += 8) {
                                const int8_t* wei_ptr = packed_wei_1x1_mmla8_.data() +
                                                        (oc / 8) * packed_wei_1x1_mmla8_stride_;
                                int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                if (use_mmla_s8) {
                                    const int8_t* src_ptrs_s8[4] = {
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                    };
                                    ker_block4x8_mmla_packed_s8(src_ptrs_s8,
                                                                wei_ptr,
                                                                dst_ptr,
                                                                IC,
                                                                0,
                                                                OC * sizeof(int32_t),
                                                                0);
                                } else {
                                    ker_block4x8_mmla_packed_u8(src_ptrs_u8,
                                                                wei_ptr,
                                                                dst_ptr,
                                                                IC,
                                                                0,
                                                                OC * sizeof(int32_t),
                                                                0);
                                    }
                                    if (use_mmla_u8 && add_comp_1x1) {
                                        for (size_t r = 0; r < 4; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 8; ++c) {
                                                dst_row[c] += wei_comp_1x1_[oc + c];
                                            }
                                        }
                                    }
                                if (bias_ptr) {
                                    int32_t* dst_row0 = dst_ptr;
                                    int32_t* dst_row1 = dst_row0 + OC;
                                    int32_t* dst_row2 = dst_row1 + OC;
                                    int32_t* dst_row3 = dst_row2 + OC;
                                    const int32_t b0 = bias_ptr[oc];
                                    const int32_t b1 = bias_ptr[oc + 1];
                                    const int32_t b2 = bias_ptr[oc + 2];
                                    const int32_t b3 = bias_ptr[oc + 3];
                                    const int32_t b4 = bias_ptr[oc + 4];
                                    const int32_t b5 = bias_ptr[oc + 5];
                                    const int32_t b6 = bias_ptr[oc + 6];
                                    const int32_t b7 = bias_ptr[oc + 7];
                                    dst_row0[0] += b0;
                                    dst_row0[1] += b1;
                                    dst_row0[2] += b2;
                                    dst_row0[3] += b3;
                                    dst_row0[4] += b4;
                                    dst_row0[5] += b5;
                                    dst_row0[6] += b6;
                                    dst_row0[7] += b7;
                                    dst_row1[0] += b0;
                                    dst_row1[1] += b1;
                                    dst_row1[2] += b2;
                                    dst_row1[3] += b3;
                                    dst_row1[4] += b4;
                                    dst_row1[5] += b5;
                                    dst_row1[6] += b6;
                                    dst_row1[7] += b7;
                                    dst_row2[0] += b0;
                                    dst_row2[1] += b1;
                                    dst_row2[2] += b2;
                                    dst_row2[3] += b3;
                                    dst_row2[4] += b4;
                                    dst_row2[5] += b5;
                                    dst_row2[6] += b6;
                                    dst_row2[7] += b7;
                                    dst_row3[0] += b0;
                                    dst_row3[1] += b1;
                                    dst_row3[2] += b2;
                                    dst_row3[3] += b3;
                                    dst_row3[4] += b4;
                                    dst_row3[5] += b5;
                                    dst_row3[6] += b6;
                                    dst_row3[7] += b7;
                                }
                            }
                        }
                        for (; oc + 4 <= OC; oc += 4) {
                            const int8_t* wei_ptr = use_block4x4_mmla
                                                        ? (packed_wei_1x1_mmla4_.data() +
                                                           (oc / 4) * packed_wei_1x1_mmla4_stride_)
                                                        : (use_packed4_1x1
                                                               ? (packed_wei_1x1_dot4_.data() +
                                                                  (oc / 4) * packed_wei_1x1_dot4_stride_)
                                                               : wei.ptr<int8_t>(oc, 0, 0, 0));
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            if (in0 && in1 && in2 && in3) {
                                if (use_block4x4_mmla) {
                                    if (use_mmla_s8) {
                                        const int8_t* src_ptrs_s8[4] = {
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                        };
                                        ker_block4x4_mmla_packed_s8(src_ptrs_s8,
                                                                    wei_ptr,
                                                                    dst_ptr,
                                                                    IC,
                                                                    0,
                                                                    OC * sizeof(int32_t),
                                                                    0);
                                    } else {
                                        ker_block4x4_mmla_packed_u8(src_ptrs_u8,
                                                                    wei_ptr,
                                                                    dst_ptr,
                                                                    IC,
                                                                    0,
                                                                    OC * sizeof(int32_t),
                                                                    0);
                                    }
                                    if (use_mmla_u8 && add_comp_1x1) {
                                        for (size_t r = 0; r < 4; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            dst_row[0] += wei_comp_1x1_[oc];
                                            dst_row[1] += wei_comp_1x1_[oc + 1];
                                            dst_row[2] += wei_comp_1x1_[oc + 2];
                                            dst_row[3] += wei_comp_1x1_[oc + 3];
                                        }
                                    }
                                } else if (use_block4x4_dot) {
                                    const int8_t* src_ptrs_s8[4] = {
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                    };
                                    if (use_packed4_1x1) {
                                        ker_block4x4_dot_packed(src_ptrs_s8,
                                                                wei_ptr,
                                                                dst_ptr,
                                                                IC,
                                                                0,
                                                                OC * sizeof(int32_t),
                                                                0);
                                    } else {
                                        ker_block4x4_dot(src_ptrs_s8,
                                                         wei_ptr,
                                                         dst_ptr,
                                                         IC,
                                                         IC,
                                                         OC * sizeof(int32_t),
                                                         0);
                                    }
                                } else {
                                    if (use_packed4_1x1) {
                                        ker_block4x4_udot_packed(src_ptrs_u8,
                                                                 wei_ptr,
                                                                 dst_ptr,
                                                                 IC,
                                                                 0,
                                                                 OC * sizeof(int32_t),
                                                                 0);
                                    } else {
                                        ker_block4x4_udot(src_ptrs_u8,
                                                          wei_ptr,
                                                          dst_ptr,
                                                          IC,
                                                          IC,
                                                          OC * sizeof(int32_t),
                                                          0);
                                    }
                                    if (add_comp_1x1) {
                                        for (size_t r = 0; r < 4; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            dst_row[0] += wei_comp_1x1_[oc];
                                            dst_row[1] += wei_comp_1x1_[oc + 1];
                                            dst_row[2] += wei_comp_1x1_[oc + 2];
                                            dst_row[3] += wei_comp_1x1_[oc + 3];
                                        }
                                    }
                                }
                            } else {
                                for (size_t i = 0; i < 4; ++i) {
                                    int32_t* dst_row = dst_ptr + i * OC;
                                    if (src_ptrs_u8[i]) {
                                        if (use_dot_s8) {
                                            ker_block4_dot(reinterpret_cast<const int8_t*>(src_ptrs_u8[i]),
                                                           wei_ptr,
                                                           dst_row,
                                                           IC,
                                                           IC,
                                                           0);
                                        } else if (use_dot_u8) {
                                            ker_block4_udot(src_ptrs_u8[i], wei_ptr, dst_row, IC, IC, 0);
                                            if (add_comp_1x1) {
                                                dst_row[0] += wei_comp_1x1_[oc];
                                                dst_row[1] += wei_comp_1x1_[oc + 1];
                                                dst_row[2] += wei_comp_1x1_[oc + 2];
                                                dst_row[3] += wei_comp_1x1_[oc + 3];
                                            }
                                        } else if (src_signed) {
                                            ker_block4_s8(src_ptrs_u8[i], wei_ptr, dst_row, IC, IC, 0);
                                        } else {
                                            ker_block4_u8(src_ptrs_u8[i], wei_ptr, dst_row, IC, IC, 0);
                                        }
                                    } else {
                                        dst_row[0] = 0;
                                        dst_row[1] = 0;
                                        dst_row[2] = 0;
                                        dst_row[3] = 0;
                                    }
                                }
                            }
                            if (bias_ptr) {
                                int32_t* dst_row0 = dst_ptr;
                                int32_t* dst_row1 = dst_row0 + OC;
                                int32_t* dst_row2 = dst_row1 + OC;
                                int32_t* dst_row3 = dst_row2 + OC;
                                const int32_t b0 = bias_ptr[oc];
                                const int32_t b1 = bias_ptr[oc + 1];
                                const int32_t b2 = bias_ptr[oc + 2];
                                const int32_t b3 = bias_ptr[oc + 3];
                                dst_row0[0] += b0;
                                dst_row0[1] += b1;
                                dst_row0[2] += b2;
                                dst_row0[3] += b3;
                                dst_row1[0] += b0;
                                dst_row1[1] += b1;
                                dst_row1[2] += b2;
                                dst_row1[3] += b3;
                                dst_row2[0] += b0;
                                dst_row2[1] += b1;
                                dst_row2[2] += b2;
                                dst_row2[3] += b3;
                                dst_row3[0] += b0;
                                dst_row3[1] += b1;
                                dst_row3[2] += b2;
                                dst_row3[3] += b3;
                            }
                        }
                        for (; oc < OC; oc++) {
                            const int8_t* wei_ptr = wei.ptr<int8_t>(oc, 0, 0, 0);
                            for (size_t i = 0; i < 4; ++i) {
                                int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow + i);
                                if (src_ptrs_u8[i]) {
                                    ker(src_ptrs_u8[i], wei_ptr, dst_ptr, IC, 0);
                                    if (bias_ptr) {
                                        *dst_ptr += bias_ptr[oc];
                                    }
                                } else {
                                    *dst_ptr = bias_ptr ? bias_ptr[oc] : 0;
                                }
                            }
                        }
                    }
                }
                if ((use_block2x8_dot || use_block2x8_udot || use_block2x16_dot_interleaved ||
                     use_block2x16_udot_interleaved || use_block2x32_dot_interleaved ||
                     use_block2x32_udot_interleaved || use_block4x16_dot_interleaved ||
                     use_block4x16_udot_interleaved) &&
                    (conv_stride_w == 1) && (pad_l == 0) && (pad_r == 0) && OW >= 2) {
                    if (ih >= 0 && ih < static_cast<ptrdiff_t>(IH)) {
                        const size_t ow_start = ow;
                        const size_t ow_end4 = OW & ~static_cast<size_t>(3);
                        const size_t ow_end = OW & ~static_cast<size_t>(1);
                        const size_t src_stride_w = src.stride(3);
                        const size_t dst_stride_w = dst.stride(3);
                        const size_t dst_stride_w_bytes = dst_stride_w * sizeof(int32_t);
                        const uint8_t* src_row =
                            src.ptr<uint8_t>(n, 0, static_cast<size_t>(ih), static_cast<size_t>(0));
                        size_t ow_pair_start = ow_start;
                        const bool use_block4x16_dot_lhs_pack =
                            use_block4x16_dot_interleaved && use_block4x16_dot_interleaved_lhs;
                        const bool use_block4x16_udot_lhs_pack =
                            use_block4x16_udot_interleaved && use_block4x16_udot_interleaved_lhs;
                        if ((use_block4x16_dot_interleaved || use_block4x16_udot_interleaved) &&
                            ow_end4 > ow_start) {
                            if (use_block4x16_dot_lhs_pack || use_block4x16_udot_lhs_pack) {
                                const size_t ow_blocks = (ow_end4 - ow_start) / 4;
                                if (use_block4x16_dot_lhs_pack) {
                                    const size_t pack_bytes = packed_lhs4x16_block_stride * ow_blocks;
                                    packed_lhs4x16_s8.resize(pack_bytes);
                                    uint8_t* packed_base =
                                        reinterpret_cast<uint8_t*>(packed_lhs4x16_s8.data());
                                    for (size_t ow2 = ow_start; ow2 < ow_end4; ow2 += 4) {
                                        uint8_t* packed_lhs =
                                            packed_base + ((ow2 - ow_start) / 4) * packed_lhs4x16_block_stride;
                                        const uint8_t* src_ptr0 = src_row + ow2 * src_stride_w;
                                        const uint8_t* src_ptr1 = src_ptr0 + src_stride_w;
                                        const uint8_t* src_ptr2 = src_ptr1 + src_stride_w;
                                        const uint8_t* src_ptr3 = src_ptr2 + src_stride_w;
                                        pack_lhs_4x16_dot_interleaved4(src_ptr0,
                                                                       src_ptr1,
                                                                       src_ptr2,
                                                                       src_ptr3,
                                                                       IC,
                                                                       packed_lhs,
                                                                       packed_lhs4x16_row_stride);
                                    }
                                } else {
                                    const size_t pack_bytes = packed_lhs4x16_block_stride * ow_blocks;
                                    packed_lhs4x16_u8.resize(pack_bytes);
                                    uint8_t* packed_base = packed_lhs4x16_u8.data();
                                    for (size_t ow2 = ow_start; ow2 < ow_end4; ow2 += 4) {
                                        uint8_t* packed_lhs =
                                            packed_base + ((ow2 - ow_start) / 4) * packed_lhs4x16_block_stride;
                                        const uint8_t* src_ptr0 = src_row + ow2 * src_stride_w;
                                        const uint8_t* src_ptr1 = src_ptr0 + src_stride_w;
                                        const uint8_t* src_ptr2 = src_ptr1 + src_stride_w;
                                        const uint8_t* src_ptr3 = src_ptr2 + src_stride_w;
                                        pack_lhs_4x16_dot_interleaved4(src_ptr0,
                                                                       src_ptr1,
                                                                       src_ptr2,
                                                                       src_ptr3,
                                                                       IC,
                                                                       packed_lhs,
                                                                       packed_lhs4x16_row_stride);
                                    }
                                }
                            }
                            for (size_t oc4 = 0; oc4 + 16 <= OC; oc4 += 16) {
                                const int8_t* wei_ptr = packed_wei_1x1_dot16_interleaved_.data() +
                                                        (oc4 / 16) * packed_wei_1x1_dot16_interleaved_stride_;
                                const int32_t b0 = bias_ptr ? bias_ptr[oc4] : 0;
                                const int32_t b1 = bias_ptr ? bias_ptr[oc4 + 1] : 0;
                                const int32_t b2 = bias_ptr ? bias_ptr[oc4 + 2] : 0;
                                const int32_t b3 = bias_ptr ? bias_ptr[oc4 + 3] : 0;
                                const int32_t b4 = bias_ptr ? bias_ptr[oc4 + 4] : 0;
                                const int32_t b5 = bias_ptr ? bias_ptr[oc4 + 5] : 0;
                                const int32_t b6 = bias_ptr ? bias_ptr[oc4 + 6] : 0;
                                const int32_t b7 = bias_ptr ? bias_ptr[oc4 + 7] : 0;
                                const int32_t b8 = bias_ptr ? bias_ptr[oc4 + 8] : 0;
                                const int32_t b9 = bias_ptr ? bias_ptr[oc4 + 9] : 0;
                                const int32_t b10 = bias_ptr ? bias_ptr[oc4 + 10] : 0;
                                const int32_t b11 = bias_ptr ? bias_ptr[oc4 + 11] : 0;
                                const int32_t b12 = bias_ptr ? bias_ptr[oc4 + 12] : 0;
                                const int32_t b13 = bias_ptr ? bias_ptr[oc4 + 13] : 0;
                                const int32_t b14 = bias_ptr ? bias_ptr[oc4 + 14] : 0;
                                const int32_t b15 = bias_ptr ? bias_ptr[oc4 + 15] : 0;
                                int32_t* dst_row_base = dst.ptr<int32_t>(n, oc4, oh, 0);
                                for (size_t ow2 = ow_start; ow2 < ow_end4; ow2 += 4) {
                                    const uint8_t* src_ptr0 = src_row + ow2 * src_stride_w;
                                    const int8_t* src_ptr0_s8 = reinterpret_cast<const int8_t*>(src_ptr0);
                                    int32_t* dst_ptr = dst_row_base + ow2 * dst_stride_w;
                                    if (use_block4x16_dot_interleaved) {
                                        if (use_block4x16_dot_lhs_pack) {
                                            int8_t* packed_lhs =
                                                packed_lhs4x16_s8.data() +
                                                ((ow2 - ow_start) / 4) * packed_lhs4x16_block_stride;
                                            ker_block4x16_dot_packed_lhs_strided_interleaved(
                                                packed_lhs, wei_ptr, dst_ptr, IC, packed_lhs4x16_row_stride,
                                                dst_stride_w_bytes, 0);
                                        } else {
                                            ker_block4x16_dot_packed_strided_interleaved(src_ptr0_s8,
                                                                                         wei_ptr,
                                                                                         dst_ptr,
                                                                                         IC,
                                                                                         src_stride_w,
                                                                                         dst_stride_w_bytes,
                                                                                         0);
                                        }
                                    } else {
                                        if (use_block4x16_udot_lhs_pack) {
                                            uint8_t* packed_lhs =
                                                packed_lhs4x16_u8.data() +
                                                ((ow2 - ow_start) / 4) * packed_lhs4x16_block_stride;
                                            ker_block4x16_udot_packed_lhs_strided_interleaved(
                                                packed_lhs, wei_ptr, dst_ptr, IC, packed_lhs4x16_row_stride,
                                                dst_stride_w_bytes, 0);
                                        } else {
                                            ker_block4x16_udot_packed_strided_interleaved(src_ptr0,
                                                                                          wei_ptr,
                                                                                          dst_ptr,
                                                                                          IC,
                                                                                          src_stride_w,
                                                                                          dst_stride_w_bytes,
                                                                                          0);
                                        }
                                        if (add_comp_1x1) {
                                            for (size_t r = 0; r < 4; ++r) {
                                                int32_t* dst_row = dst_ptr + r * OC;
                                                for (size_t c = 0; c < 16; ++c) {
                                                    dst_row[c] += wei_comp_1x1_[oc4 + c];
                                                }
                                            }
                                        }
                                    }
                                    if (bias_ptr) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        int32_t* dst_row2 = dst_row1 + OC;
                                        int32_t* dst_row3 = dst_row2 + OC;
                                        dst_row0[0] += b0;
                                        dst_row0[1] += b1;
                                        dst_row0[2] += b2;
                                        dst_row0[3] += b3;
                                        dst_row0[4] += b4;
                                        dst_row0[5] += b5;
                                        dst_row0[6] += b6;
                                        dst_row0[7] += b7;
                                        dst_row0[8] += b8;
                                        dst_row0[9] += b9;
                                        dst_row0[10] += b10;
                                        dst_row0[11] += b11;
                                        dst_row0[12] += b12;
                                        dst_row0[13] += b13;
                                        dst_row0[14] += b14;
                                        dst_row0[15] += b15;
                                        dst_row1[0] += b0;
                                        dst_row1[1] += b1;
                                        dst_row1[2] += b2;
                                        dst_row1[3] += b3;
                                        dst_row1[4] += b4;
                                        dst_row1[5] += b5;
                                        dst_row1[6] += b6;
                                        dst_row1[7] += b7;
                                        dst_row1[8] += b8;
                                        dst_row1[9] += b9;
                                        dst_row1[10] += b10;
                                        dst_row1[11] += b11;
                                        dst_row1[12] += b12;
                                        dst_row1[13] += b13;
                                        dst_row1[14] += b14;
                                        dst_row1[15] += b15;
                                        dst_row2[0] += b0;
                                        dst_row2[1] += b1;
                                        dst_row2[2] += b2;
                                        dst_row2[3] += b3;
                                        dst_row2[4] += b4;
                                        dst_row2[5] += b5;
                                        dst_row2[6] += b6;
                                        dst_row2[7] += b7;
                                        dst_row2[8] += b8;
                                        dst_row2[9] += b9;
                                        dst_row2[10] += b10;
                                        dst_row2[11] += b11;
                                        dst_row2[12] += b12;
                                        dst_row2[13] += b13;
                                        dst_row2[14] += b14;
                                        dst_row2[15] += b15;
                                        dst_row3[0] += b0;
                                        dst_row3[1] += b1;
                                        dst_row3[2] += b2;
                                        dst_row3[3] += b3;
                                        dst_row3[4] += b4;
                                        dst_row3[5] += b5;
                                        dst_row3[6] += b6;
                                        dst_row3[7] += b7;
                                        dst_row3[8] += b8;
                                        dst_row3[9] += b9;
                                        dst_row3[10] += b10;
                                        dst_row3[11] += b11;
                                        dst_row3[12] += b12;
                                        dst_row3[13] += b13;
                                        dst_row3[14] += b14;
                                        dst_row3[15] += b15;
                                    }
                                }
                            }
                            ow_pair_start = ow_end4;
                        }
                        size_t oc = 0;
                        if (use_block2x32_dot_interleaved || use_block2x32_udot_interleaved) {
                            for (; oc + 32 <= OC; oc += 32) {
                                const int8_t* wei_ptr = packed_wei_1x1_dot32_interleaved_.data() +
                                                        (oc / 32) * packed_wei_1x1_dot32_interleaved_stride_;
                                const int32_t* bias_block = bias_ptr ? bias_ptr + oc : nullptr;
                                int32_t* dst_row_base = dst.ptr<int32_t>(n, oc, oh, 0);
                                for (size_t ow2 = ow_pair_start; ow2 < ow_end; ow2 += 2) {
                                    const uint8_t* src_ptr0 = src_row + ow2 * src_stride_w;
                                    const int8_t* src_ptr0_s8 = reinterpret_cast<const int8_t*>(src_ptr0);
                                    int32_t* dst_ptr = dst_row_base + ow2 * dst_stride_w;
                                    if (use_block2x32_dot_interleaved) {
                                        ker_block2x32_dot_packed_strided_interleaved(src_ptr0_s8,
                                                                                     wei_ptr,
                                                                                     dst_ptr,
                                                                                     IC,
                                                                                     src_stride_w,
                                                                                     dst_stride_w_bytes,
                                                                                     0);
                                    } else {
                                        ker_block2x32_udot_packed_strided_interleaved(src_ptr0,
                                                                                      wei_ptr,
                                                                                      dst_ptr,
                                                                                      IC,
                                                                                      src_stride_w,
                                                                                      dst_stride_w_bytes,
                                                                                      nullptr,
                                                                                      0);
                                        if (add_comp_1x1) {
                                            add_1x1_block_2x32(dst_ptr, OC, wei_comp_1x1_.data() + oc);
                                        }
                                    }
                                    if (bias_block && !use_bias_comp_1x1) {
                                        add_1x1_block_2x32(dst_ptr, OC, bias_block);
                                    }
                                }
                            }
                        }
                        if (use_block2x16_dot_interleaved || use_block2x16_udot_interleaved) {
                            for (; oc + 16 <= OC; oc += 16) {
                                const int8_t* wei_ptr = packed_wei_1x1_dot16_interleaved_.data() +
                                                        (oc / 16) * packed_wei_1x1_dot16_interleaved_stride_;
                                const int32_t b0 = bias_ptr ? bias_ptr[oc] : 0;
                                const int32_t b1 = bias_ptr ? bias_ptr[oc + 1] : 0;
                                const int32_t b2 = bias_ptr ? bias_ptr[oc + 2] : 0;
                                const int32_t b3 = bias_ptr ? bias_ptr[oc + 3] : 0;
                                const int32_t b4 = bias_ptr ? bias_ptr[oc + 4] : 0;
                                const int32_t b5 = bias_ptr ? bias_ptr[oc + 5] : 0;
                                const int32_t b6 = bias_ptr ? bias_ptr[oc + 6] : 0;
                                const int32_t b7 = bias_ptr ? bias_ptr[oc + 7] : 0;
                                const int32_t b8 = bias_ptr ? bias_ptr[oc + 8] : 0;
                                const int32_t b9 = bias_ptr ? bias_ptr[oc + 9] : 0;
                                const int32_t b10 = bias_ptr ? bias_ptr[oc + 10] : 0;
                                const int32_t b11 = bias_ptr ? bias_ptr[oc + 11] : 0;
                                const int32_t b12 = bias_ptr ? bias_ptr[oc + 12] : 0;
                                const int32_t b13 = bias_ptr ? bias_ptr[oc + 13] : 0;
                                const int32_t b14 = bias_ptr ? bias_ptr[oc + 14] : 0;
                                const int32_t b15 = bias_ptr ? bias_ptr[oc + 15] : 0;
                                int32_t* dst_row_base = dst.ptr<int32_t>(n, oc, oh, 0);
                                for (size_t ow2 = ow_pair_start; ow2 < ow_end; ow2 += 2) {
                                    const uint8_t* src_ptr0 = src_row + ow2 * src_stride_w;
                                    const int8_t* src_ptr0_s8 = reinterpret_cast<const int8_t*>(src_ptr0);
                                    int32_t* dst_ptr = dst_row_base + ow2 * dst_stride_w;
                                    if (use_block2x16_dot_interleaved) {
                                        ker_block2x16_dot_packed_strided_interleaved(src_ptr0_s8,
                                                                                     wei_ptr,
                                                                                     dst_ptr,
                                                                                     IC,
                                                                                     src_stride_w,
                                                                                     dst_stride_w_bytes,
                                                                                     0);
                                    } else {
                                        ker_block2x16_udot_packed_strided_interleaved(src_ptr0,
                                                                                      wei_ptr,
                                                                                      dst_ptr,
                                                                                      IC,
                                                                                      src_stride_w,
                                                                                      dst_stride_w_bytes,
                                                                                      0);
                                        if (add_comp_1x1) {
                                            add_1x1_block_2x16(dst_ptr, OC, wei_comp_1x1_.data() + oc);
                                        }
                                    }
                                    if (bias_ptr) {
                                        int32_t* dst_row0 = dst_ptr;
                                        int32_t* dst_row1 = dst_row0 + OC;
                                        dst_row0[0] += b0;
                                        dst_row0[1] += b1;
                                        dst_row0[2] += b2;
                                        dst_row0[3] += b3;
                                        dst_row0[4] += b4;
                                        dst_row0[5] += b5;
                                        dst_row0[6] += b6;
                                        dst_row0[7] += b7;
                                        dst_row0[8] += b8;
                                        dst_row0[9] += b9;
                                        dst_row0[10] += b10;
                                        dst_row0[11] += b11;
                                        dst_row0[12] += b12;
                                        dst_row0[13] += b13;
                                        dst_row0[14] += b14;
                                        dst_row0[15] += b15;
                                        dst_row1[0] += b0;
                                        dst_row1[1] += b1;
                                        dst_row1[2] += b2;
                                        dst_row1[3] += b3;
                                        dst_row1[4] += b4;
                                        dst_row1[5] += b5;
                                        dst_row1[6] += b6;
                                        dst_row1[7] += b7;
                                        dst_row1[8] += b8;
                                        dst_row1[9] += b9;
                                        dst_row1[10] += b10;
                                        dst_row1[11] += b11;
                                        dst_row1[12] += b12;
                                        dst_row1[13] += b13;
                                        dst_row1[14] += b14;
                                        dst_row1[15] += b15;
                                    }
                                }
                            }
                        }
                        for (; oc + 8 <= OC; oc += 8) {
                            const bool use_interleaved =
                                use_block2x8_dot_interleaved || use_block2x8_udot_interleaved;
                            const int8_t* wei_ptr =
                                use_interleaved
                                    ? (packed_wei_1x1_dot8_interleaved_.data() +
                                       (oc / 8) * packed_wei_1x1_dot8_interleaved_stride_)
                                    : (packed_wei_1x1_dot8_.data() + (oc / 8) * packed_wei_1x1_dot8_stride_);
                            const int32_t b0 = bias_ptr ? bias_ptr[oc] : 0;
                            const int32_t b1 = bias_ptr ? bias_ptr[oc + 1] : 0;
                            const int32_t b2 = bias_ptr ? bias_ptr[oc + 2] : 0;
                            const int32_t b3 = bias_ptr ? bias_ptr[oc + 3] : 0;
                            const int32_t b4 = bias_ptr ? bias_ptr[oc + 4] : 0;
                            const int32_t b5 = bias_ptr ? bias_ptr[oc + 5] : 0;
                            const int32_t b6 = bias_ptr ? bias_ptr[oc + 6] : 0;
                            const int32_t b7 = bias_ptr ? bias_ptr[oc + 7] : 0;
                            int32_t* dst_row_base = dst.ptr<int32_t>(n, oc, oh, 0);
                            for (size_t ow2 = ow_pair_start; ow2 < ow_end; ow2 += 2) {
                                const uint8_t* src_ptr0 = src_row + ow2 * src_stride_w;
                                const int8_t* src_ptr0_s8 = reinterpret_cast<const int8_t*>(src_ptr0);
                                int32_t* dst_ptr = dst_row_base + ow2 * dst_stride_w;
                                if (use_block2x8_dot_interleaved) {
                                    ker_block2x8_dot_packed_strided_interleaved(src_ptr0_s8,
                                                                                wei_ptr,
                                                                                dst_ptr,
                                                                                IC,
                                                                                src_stride_w,
                                                                                dst_stride_w_bytes,
                                                                                0);
                                } else if (use_block2x8_dot) {
                                    ker_block2x8_dot_packed_strided(src_ptr0_s8,
                                                                    wei_ptr,
                                                                    dst_ptr,
                                                                    IC,
                                                                    src_stride_w,
                                                                    dst_stride_w_bytes,
                                                                    0);
                                } else if (use_block2x8_udot_interleaved) {
                                    ker_block2x8_udot_packed_strided_interleaved(src_ptr0,
                                                                                 wei_ptr,
                                                                                 dst_ptr,
                                                                                 IC,
                                                                                 src_stride_w,
                                                                                 dst_stride_w_bytes,
                                                                                 0);
                                } else {
                                    ker_block2x8_udot_packed_strided(src_ptr0,
                                                                     wei_ptr,
                                                                     dst_ptr,
                                                                     IC,
                                                                     src_stride_w,
                                                                     dst_stride_w_bytes,
                                                                     0);
                                    if (add_comp_1x1) {
                                        for (size_t r = 0; r < 2; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 8; ++c) {
                                                dst_row[c] += wei_comp_1x1_[oc + c];
                                            }
                                        }
                                    }
                                }
                                if (bias_ptr) {
                                    int32_t* dst_row0 = dst_ptr;
                                    int32_t* dst_row1 = dst_row0 + OC;
                                    dst_row0[0] += b0;
                                    dst_row0[1] += b1;
                                    dst_row0[2] += b2;
                                    dst_row0[3] += b3;
                                    dst_row0[4] += b4;
                                    dst_row0[5] += b5;
                                    dst_row0[6] += b6;
                                    dst_row0[7] += b7;
                                    dst_row1[0] += b0;
                                    dst_row1[1] += b1;
                                    dst_row1[2] += b2;
                                    dst_row1[3] += b3;
                                    dst_row1[4] += b4;
                                    dst_row1[5] += b5;
                                    dst_row1[6] += b6;
                                    dst_row1[7] += b7;
                                }
                            }
                        }
                        ow = ow_end;
                    }
                }
                for (; ow < OW; ow++) {
                    const ptrdiff_t iw = static_cast<ptrdiff_t>(ow * conv_stride_w) - pad_l;
                    const bool in_bounds = ih >= 0 && ih < static_cast<ptrdiff_t>(IH) && iw >= 0 &&
                                           iw < static_cast<ptrdiff_t>(IW);
                    const uint8_t* src_ptr =
                        in_bounds ? src.ptr<uint8_t>(n, 0, static_cast<size_t>(ih), static_cast<size_t>(iw)) : nullptr;
                    size_t oc = 0;
                    if (use_dot_s8) {
                        for (; oc + 8 <= OC; oc += 8) {
                            const int8_t* wei_ptr =
                                use_packed8_1x1
                                    ? (packed_wei_1x1_dot8_.data() + (oc / 8) * packed_wei_1x1_dot8_stride_)
                                    : wei.ptr<int8_t>(oc, 0, 0, 0);
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            if (src_ptr) {
                                const int8_t* src_ptr_s8 = reinterpret_cast<const int8_t*>(src_ptr);
                                if (use_packed8_1x1) {
                                    ker_block8_dot_packed(src_ptr_s8, wei_ptr, dst_ptr, IC, 0, 0);
                                } else {
                                    ker_block8_dot(src_ptr_s8, wei_ptr, dst_ptr, IC, IC, 0);
                                }
                                if (bias_ptr) {
                                    for (size_t i = 0; i < 8; ++i) {
                                        dst_ptr[i] += bias_ptr[oc + i];
                                    }
                                }
                            } else {
                                for (size_t i = 0; i < 8; ++i) {
                                    dst_ptr[i] = bias_ptr ? bias_ptr[oc + i] : 0;
                                }
                            }
                        }
                    } else if (use_dot_u8) {
                        for (; oc + 8 <= OC; oc += 8) {
                            const int8_t* wei_ptr =
                                use_packed8_1x1
                                    ? (packed_wei_1x1_dot8_.data() + (oc / 8) * packed_wei_1x1_dot8_stride_)
                                    : wei.ptr<int8_t>(oc, 0, 0, 0);
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            if (src_ptr) {
                                if (use_packed8_1x1) {
                                    ker_block8_udot_packed(src_ptr, wei_ptr, dst_ptr, IC, 0, 0);
                                } else {
                                    ker_block8_udot(src_ptr, wei_ptr, dst_ptr, IC, IC, 0);
                                }
                                if (add_comp_1x1) {
                                    for (size_t i = 0; i < 8; ++i) {
                                        dst_ptr[i] += wei_comp_1x1_[oc + i];
                                    }
                                }
                                if (bias_ptr) {
                                    for (size_t i = 0; i < 8; ++i) {
                                        dst_ptr[i] += bias_ptr[oc + i];
                                    }
                                }
                            } else {
                                for (size_t i = 0; i < 8; ++i) {
                                    dst_ptr[i] = bias_ptr ? bias_ptr[oc + i] : 0;
                                }
                            }
                        }
                    }
                    for (; oc + 4 <= OC; oc += 4) {
                        const int8_t* wei_ptr = wei.ptr<int8_t>(oc, 0, 0, 0);
                        int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                        if (src_ptr) {
                            if (use_dot_s8) {
                                const int8_t* src_ptr_s8 = reinterpret_cast<const int8_t*>(src_ptr);
                                ker_block4_dot(src_ptr_s8, wei_ptr, dst_ptr, IC, IC, 0);
                            } else if (use_dot_u8) {
                                ker_block4_udot(src_ptr, wei_ptr, dst_ptr, IC, IC, 0);
                                if (add_comp_1x1) {
                                    dst_ptr[0] += wei_comp_1x1_[oc];
                                    dst_ptr[1] += wei_comp_1x1_[oc + 1];
                                    dst_ptr[2] += wei_comp_1x1_[oc + 2];
                                    dst_ptr[3] += wei_comp_1x1_[oc + 3];
                                }
                            } else if (src_signed) {
                                ker_block4_s8(src_ptr, wei_ptr, dst_ptr, IC, IC, 0);
                            } else {
                                ker_block4_u8(src_ptr, wei_ptr, dst_ptr, IC, IC, 0);
                            }
                            if (bias_ptr) {
                                dst_ptr[0] += bias_ptr[oc];
                                dst_ptr[1] += bias_ptr[oc + 1];
                                dst_ptr[2] += bias_ptr[oc + 2];
                                dst_ptr[3] += bias_ptr[oc + 3];
                            }
                        } else {
                            dst_ptr[0] = bias_ptr ? bias_ptr[oc] : 0;
                            dst_ptr[1] = bias_ptr ? bias_ptr[oc + 1] : 0;
                            dst_ptr[2] = bias_ptr ? bias_ptr[oc + 2] : 0;
                            dst_ptr[3] = bias_ptr ? bias_ptr[oc + 3] : 0;
                        }
                    }
                    for (; oc < OC; oc++) {
                        const int8_t* wei_ptr = wei.ptr<int8_t>(oc, 0, 0, 0);
                        int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                        if (src_ptr) {
                            ker(src_ptr, wei_ptr, dst_ptr, IC, 0);
                            if (bias_ptr) {
                                *dst_ptr += bias_ptr[oc];
                            }
                        } else {
                            *dst_ptr = bias_ptr ? bias_ptr[oc] : 0;
                        }
                    }
                }
            }
        }
    } else {
        const size_t stride_h = src.stride(2);
        const size_t stride_w = src.stride(3);
        const bool use_brgemm = !packed_wei_brgemm_.empty();
        const bool use_packed4_kxk = has_dotprod_ && !packed_wei_brgemm_dot4_.empty();
        const bool use_packed8_kxk = has_dotprod_ && !packed_wei_brgemm_dot8_.empty();
        const bool use_packed16_kxk_mmla =
            (use_mmla_s8 || use_mmla_u8) && (IC % 8 == 0) && !packed_wei_brgemm_mmla16_.empty();
        const bool use_packed8_kxk_mmla =
            (use_mmla_s8 || use_mmla_u8) && (IC % 8 == 0) && !packed_wei_brgemm_mmla8_.empty();
        const bool use_packed4_kxk_mmla =
            (use_mmla_s8 || use_mmla_u8) && (IC % 8 == 0) && !packed_wei_brgemm_mmla4_.empty();
        const size_t kxk_k = IC * KH * KW;
        const size_t kxk_kp = round_up(kxk_k, 8);
        const bool use_packed16_kxk_mmla_fused =
            (use_mmla_s8 || use_mmla_u8) && (packed_wei_brgemm_mmla_fused_k_ == kxk_kp) &&
            !packed_wei_brgemm_mmla16_fused_.empty();
        const bool use_packed12_kxk_mmla_fused =
            (use_mmla_s8 || use_mmla_u8) && (packed_wei_brgemm_mmla_fused_k_ == kxk_kp) &&
            !packed_wei_brgemm_mmla12_fused_.empty();
        const bool use_packed8_kxk_mmla_fused =
            (use_mmla_s8 || use_mmla_u8) && (packed_wei_brgemm_mmla_fused_k_ == kxk_kp) &&
            !packed_wei_brgemm_mmla8_fused_.empty();
        const bool use_packed4_kxk_mmla_fused =
            (use_mmla_s8 || use_mmla_u8) && (packed_wei_brgemm_mmla_fused_k_ == kxk_kp) &&
            !packed_wei_brgemm_mmla4_fused_.empty();
        const size_t oc_blocks4 = use_packed4_kxk ? (OC / 4) : 0;
        const size_t oc_blocks4_mmla = use_packed4_kxk_mmla ? (OC / 4) : 0;
        const size_t oc_blocks8 = use_packed8_kxk ? (OC / 8) : 0;
        const size_t oc_blocks8_mmla = use_packed8_kxk_mmla ? (OC / 8) : 0;
        const size_t oc_blocks16_mmla = use_packed16_kxk_mmla ? (OC / 16) : 0;
        const bool use_kxk_mmla_fused = use_packed16_kxk_mmla_fused || use_packed12_kxk_mmla_fused ||
                                        use_packed8_kxk_mmla_fused || use_packed4_kxk_mmla_fused;
        const size_t lda_brgemm = stride_w * conv_stride_w;
        const size_t ldc_brgemm = dst.stride(3);

        const size_t kxk_m = OH * OW;
        const bool use_kxk_mmla_im2col =
            nhwc_contiguous &&
            (((kxk_m <= 1024) && (conv_stride_h == 1) && (conv_stride_w == 1) &&
              (pad_l == 0) && (pad_r == 0) && (pad_t == 0) && (pad_b == 0)) ||
             ((KH * KW) <= 9 && (kxk_m <= 4096))) &&
            (use_mmla_s8 || use_mmla_u8) && (IC % 8 == 0) && (kxk_k % 8 == 0) &&
            (packed_wei_brgemm_mmla_fused_k_ == kxk_kp) &&
            (!packed_wei_brgemm_mmla16_fused_.empty() || !packed_wei_brgemm_mmla12_fused_.empty() ||
             !packed_wei_brgemm_mmla8_fused_.empty() || !packed_wei_brgemm_mmla4_fused_.empty());
        if (use_kxk_mmla_im2col) {
            const bool use_block12_strategy =
                !packed_wei_brgemm_mmla12_fused_.empty() &&
                ((OC % 12) == 0 || (OC % 12) == 4 || (OC % 12) == 8);
            const bool use_block16 = !packed_wei_brgemm_mmla16_fused_.empty();
            const bool use_block8 = !packed_wei_brgemm_mmla8_fused_.empty();
            const bool use_block4 = !packed_wei_brgemm_mmla4_fused_.empty();
            const size_t dst_row_stride = OC;
            const size_t dst_row_stride_bytes = dst_row_stride * sizeof(int32_t);
            size_t kIm2ColMB = (128 * 1024) / std::max<size_t>(1, kxk_k);
            kIm2ColMB = (kIm2ColMB / 8) * 8;
            kIm2ColMB = std::max<size_t>(32, std::min<size_t>(256, kIm2ColMB));
            const size_t m_blocks = (kxk_m + kIm2ColMB - 1) / kIm2ColMB;
            const bool has_comp_u8 = use_mmla_u8 && !wei_comp_brgemm_.empty();
            const bool has_bias = bias_ptr != nullptr;
            const bool use_row_prefill = has_comp_u8 || has_bias;
            const uint64_t kernel_accum = use_row_prefill ? 1 : 0;

            auto add_comp_bias = [&](int32_t* dst_ptr, size_t rows, size_t cols, size_t oc_base) {
                if (use_row_prefill) {
                    return;
                }
                if (has_comp_u8) {
                    for (size_t r = 0; r < rows; ++r) {
                        int32_t* dst_row = dst_ptr + r * dst_row_stride;
                        for (size_t c = 0; c < cols; ++c) {
                            dst_row[c] += wei_comp_brgemm_[oc_base + c];
                        }
                    }
                }
                if (has_bias) {
                    for (size_t r = 0; r < rows; ++r) {
                        int32_t* dst_row = dst_ptr + r * dst_row_stride;
                        for (size_t c = 0; c < cols; ++c) {
                            dst_row[c] += bias_ptr[oc_base + c];
                        }
                    }
                }
            };

            ov::parallel_for2d(N, m_blocks, [&](size_t n, size_t mb_idx) {
                static thread_local std::vector<uint8_t> im2col_u8;
                static thread_local std::vector<int8_t> im2col_s8;
                const size_t m_start = mb_idx * kIm2ColMB;
                if (m_start >= kxk_m) {
                    return;
                }
                const size_t m_end = std::min(m_start + kIm2ColMB, kxk_m);
                const size_t m_block = m_end - m_start;
                if (src_signed) {
                    im2col_s8.resize(m_block * kxk_k);
                } else {
                    im2col_u8.resize(m_block * kxk_k);
                }

                const uint8_t* src_base = src.ptr<uint8_t>(n, 0, 0, 0);
                int32_t* dst_base = dst.ptr<int32_t>(n, 0, 0, 0);
                auto pack_im2col_row = [&](size_t oh, size_t ow, uint8_t* dst_row_bytes) {
                    const ptrdiff_t ih0 = static_cast<ptrdiff_t>(oh * conv_stride_h) - pad_t;
                    const ptrdiff_t iw0 = static_cast<ptrdiff_t>(ow * conv_stride_w) - pad_l;
                    const bool fast_contig =
                        ih0 >= 0 && iw0 >= 0 &&
                        (ih0 + static_cast<ptrdiff_t>(KH)) <= static_cast<ptrdiff_t>(IH) &&
                        (iw0 + static_cast<ptrdiff_t>(KW)) <= static_cast<ptrdiff_t>(IW);

                    size_t col = 0;
                    if (fast_contig) {
                        const size_t row_bytes = KW * IC;
                        const size_t ih_base = static_cast<size_t>(ih0);
                        const size_t iw_base = static_cast<size_t>(iw0);
                        for (size_t kh = 0; kh < KH; ++kh) {
                            const uint8_t* src_h = src_base + (ih_base + kh) * stride_h + iw_base * stride_w;
                            std::memcpy(dst_row_bytes + col, src_h, row_bytes);
                            col += row_bytes;
                        }
                        return;
                    }

                    for (size_t kh = 0; kh < KH; ++kh) {
                        uint8_t* out_row = dst_row_bytes + col;
                        const ptrdiff_t ih = ih0 + static_cast<ptrdiff_t>(kh);
                        if (ih < 0 || ih >= static_cast<ptrdiff_t>(IH)) {
                            std::memset(out_row, 0, KW * IC);
                            col += KW * IC;
                            continue;
                        }

                        ptrdiff_t left = (iw0 < 0) ? -iw0 : 0;
                        ptrdiff_t right = 0;
                        const ptrdiff_t iw_end = iw0 + static_cast<ptrdiff_t>(KW);
                        if (iw_end > static_cast<ptrdiff_t>(IW)) {
                            right = iw_end - static_cast<ptrdiff_t>(IW);
                        }
                        if (left > static_cast<ptrdiff_t>(KW)) {
                            left = static_cast<ptrdiff_t>(KW);
                        }
                        if (right > static_cast<ptrdiff_t>(KW)) {
                            right = static_cast<ptrdiff_t>(KW);
                        }

                        const size_t valid = KW - static_cast<size_t>(left) - static_cast<size_t>(right);
                        if (left) {
                            std::memset(out_row, 0, static_cast<size_t>(left) * IC);
                        }
                        if (valid) {
                            const size_t iw_src = static_cast<size_t>(std::max<ptrdiff_t>(0, iw0));
                            const uint8_t* src_h = src_base + static_cast<size_t>(ih) * stride_h + iw_src * stride_w;
                            std::memcpy(out_row + static_cast<size_t>(left) * IC, src_h, valid * IC);
                        }
                        if (right) {
                            std::memset(out_row + (static_cast<size_t>(left) + valid) * IC,
                                        0,
                                        static_cast<size_t>(right) * IC);
                        }
                        col += KW * IC;
                    }
                };
                for (size_t r = 0; r < m_block; ++r) {
                    const size_t m = m_start + r;
                    const size_t oh = m / OW;
                    const size_t ow = m - oh * OW;
                    if (src_signed) {
                        pack_im2col_row(oh, ow, reinterpret_cast<uint8_t*>(im2col_s8.data() + r * kxk_k));
                    } else {
                        pack_im2col_row(oh, ow, im2col_u8.data() + r * kxk_k);
                    }
                }
                if (use_row_prefill) {
                    for (size_t r = 0; r < m_block; ++r) {
                        int32_t* dst_row = dst_base + (m_start + r) * OC;
                        if (has_comp_u8 && has_bias) {
                            for (size_t oc = 0; oc < OC; ++oc) {
                                dst_row[oc] = wei_comp_brgemm_[oc] + bias_ptr[oc];
                            }
                        } else if (has_comp_u8) {
                            for (size_t oc = 0; oc < OC; ++oc) {
                                dst_row[oc] = wei_comp_brgemm_[oc];
                            }
                        } else {
                            std::memcpy(dst_row, bias_ptr, OC * sizeof(int32_t));
                        }
                    }
                }

                size_t m = 0;
                for (; m + 8 <= m_block; m += 8) {
                    const int8_t* a_ptrs_s8[8] = {
                        im2col_s8.data() + (m + 0) * kxk_k,
                        im2col_s8.data() + (m + 1) * kxk_k,
                        im2col_s8.data() + (m + 2) * kxk_k,
                        im2col_s8.data() + (m + 3) * kxk_k,
                        im2col_s8.data() + (m + 4) * kxk_k,
                        im2col_s8.data() + (m + 5) * kxk_k,
                        im2col_s8.data() + (m + 6) * kxk_k,
                        im2col_s8.data() + (m + 7) * kxk_k,
                    };
                    const uint8_t* a_ptrs_u8[8] = {
                        im2col_u8.data() + (m + 0) * kxk_k,
                        im2col_u8.data() + (m + 1) * kxk_k,
                        im2col_u8.data() + (m + 2) * kxk_k,
                        im2col_u8.data() + (m + 3) * kxk_k,
                        im2col_u8.data() + (m + 4) * kxk_k,
                        im2col_u8.data() + (m + 5) * kxk_k,
                        im2col_u8.data() + (m + 6) * kxk_k,
                        im2col_u8.data() + (m + 7) * kxk_k,
                    };
                    const int8_t* a_ptrs_s8_lo[4] = {a_ptrs_s8[0], a_ptrs_s8[1], a_ptrs_s8[2], a_ptrs_s8[3]};
                    const int8_t* a_ptrs_s8_hi[4] = {a_ptrs_s8[4], a_ptrs_s8[5], a_ptrs_s8[6], a_ptrs_s8[7]};
                    const uint8_t* a_ptrs_u8_lo[4] = {a_ptrs_u8[0], a_ptrs_u8[1], a_ptrs_u8[2], a_ptrs_u8[3]};
                    const uint8_t* a_ptrs_u8_hi[4] = {a_ptrs_u8[4], a_ptrs_u8[5], a_ptrs_u8[6], a_ptrs_u8[7]};

                    if (use_block12_strategy) {
                        const size_t n12_main = (OC / 12) * 12;
                        const size_t rem12 = OC - n12_main;
                        for (size_t oc = 0; oc + 12 <= n12_main; oc += 12) {
                            const int8_t* wei_ptr =
                                packed_wei_brgemm_mmla12_fused_.data() +
                                (oc / 12) * packed_wei_brgemm_mmla12_fused_stride_;
                            int32_t* dst_ptr = dst_base + (m_start + m) * OC + oc;
                            if (use_mmla_s8) {
                                ker_block8x12_mmla_packed_s8(a_ptrs_s8, wei_ptr, dst_ptr, kxk_k, 0,
                                                            dst_row_stride_bytes, kernel_accum);
                            } else {
                                ker_block8x12_mmla_packed_u8(a_ptrs_u8, wei_ptr, dst_ptr, kxk_k, 0,
                                                            dst_row_stride_bytes, kernel_accum);
                            }
                            add_comp_bias(dst_ptr, 8, 12, oc);
                        }
                        if (rem12 >= 8) {
                            const size_t oc = n12_main;
                            if (use_block8) {
                                const int8_t* wei_ptr =
                                    packed_wei_brgemm_mmla8_fused_.data() +
                                    (oc / 8) * packed_wei_brgemm_mmla8_fused_stride_;
                                int32_t* dst_ptr = dst_base + (m_start + m) * OC + oc;
                                if (use_mmla_s8) {
                                    ker_block8x8_mmla_packed_s8(a_ptrs_s8, wei_ptr, dst_ptr, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                } else {
                                    ker_block8x8_mmla_packed_u8(a_ptrs_u8, wei_ptr, dst_ptr, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                }
                                add_comp_bias(dst_ptr, 8, 8, oc);
                            } else if (use_block4) {
                                const int8_t* wei_ptr0 =
                                    packed_wei_brgemm_mmla4_fused_.data() +
                                    (oc / 4) * packed_wei_brgemm_mmla4_fused_stride_;
                                const int8_t* wei_ptr1 =
                                    packed_wei_brgemm_mmla4_fused_.data() +
                                    ((oc + 4) / 4) * packed_wei_brgemm_mmla4_fused_stride_;
                                int32_t* dst_ptr0 = dst_base + (m_start + m) * OC + oc;
                                int32_t* dst_ptr1 = dst_ptr0 + 4;
                                int32_t* dst_ptr2 = dst_ptr0 + 4 * OC;
                                int32_t* dst_ptr3 = dst_ptr2 + 4;
                                if (use_mmla_s8) {
                                    ker_block4x4_mmla_packed_s8(a_ptrs_s8_lo, wei_ptr0, dst_ptr0, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                    ker_block4x4_mmla_packed_s8(a_ptrs_s8_lo, wei_ptr1, dst_ptr1, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                    ker_block4x4_mmla_packed_s8(a_ptrs_s8_hi, wei_ptr0, dst_ptr2, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                    ker_block4x4_mmla_packed_s8(a_ptrs_s8_hi, wei_ptr1, dst_ptr3, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                } else {
                                    ker_block4x4_mmla_packed_u8(a_ptrs_u8_lo, wei_ptr0, dst_ptr0, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                    ker_block4x4_mmla_packed_u8(a_ptrs_u8_lo, wei_ptr1, dst_ptr1, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                    ker_block4x4_mmla_packed_u8(a_ptrs_u8_hi, wei_ptr0, dst_ptr2, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                    ker_block4x4_mmla_packed_u8(a_ptrs_u8_hi, wei_ptr1, dst_ptr3, kxk_k, 0,
                                                               dst_row_stride_bytes, kernel_accum);
                                }
                                add_comp_bias(dst_ptr0, 4, 4, oc);
                                add_comp_bias(dst_ptr1, 4, 4, oc + 4);
                                add_comp_bias(dst_ptr2, 4, 4, oc);
                                add_comp_bias(dst_ptr3, 4, 4, oc + 4);
                            }
                        }
                        if (rem12 == 4) {
                            const size_t oc = n12_main;
                            const int8_t* wei_ptr =
                                packed_wei_brgemm_mmla4_fused_.data() +
                                (oc / 4) * packed_wei_brgemm_mmla4_fused_stride_;
                            int32_t* dst_ptr0 = dst_base + (m_start + m) * OC + oc;
                            int32_t* dst_ptr1 = dst_ptr0 + 4 * OC;
                            if (use_mmla_s8) {
                                ker_block4x4_mmla_packed_s8(a_ptrs_s8_lo, wei_ptr, dst_ptr0, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                                ker_block4x4_mmla_packed_s8(a_ptrs_s8_hi, wei_ptr, dst_ptr1, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                            } else {
                                ker_block4x4_mmla_packed_u8(a_ptrs_u8_lo, wei_ptr, dst_ptr0, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                                ker_block4x4_mmla_packed_u8(a_ptrs_u8_hi, wei_ptr, dst_ptr1, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                            }
                            add_comp_bias(dst_ptr0, 4, 4, oc);
                            add_comp_bias(dst_ptr1, 4, 4, oc);
                        }
                    } else {
                        for (size_t oc = 0; use_block8 && oc + 8 <= OC; oc += 8) {
                            const int8_t* wei_ptr =
                                packed_wei_brgemm_mmla8_fused_.data() +
                                (oc / 8) * packed_wei_brgemm_mmla8_fused_stride_;
                            int32_t* dst_ptr = dst_base + (m_start + m) * OC + oc;
                            if (use_mmla_s8) {
                                ker_block8x8_mmla_packed_s8(a_ptrs_s8, wei_ptr, dst_ptr, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                            } else {
                                ker_block8x8_mmla_packed_u8(a_ptrs_u8, wei_ptr, dst_ptr, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                            }
                            add_comp_bias(dst_ptr, 8, 8, oc);
                        }
                        if (use_block4 && (OC % 8) >= 4) {
                            const size_t oc_tail = (OC / 8) * 8;
                            const int8_t* wei_ptr =
                                packed_wei_brgemm_mmla4_fused_.data() +
                                (oc_tail / 4) * packed_wei_brgemm_mmla4_fused_stride_;
                            int32_t* dst_ptr0 = dst_base + (m_start + m) * OC + oc_tail;
                            int32_t* dst_ptr1 = dst_ptr0 + 4 * OC;
                            if (use_mmla_s8) {
                                ker_block4x4_mmla_packed_s8(a_ptrs_s8_lo, wei_ptr, dst_ptr0, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                                ker_block4x4_mmla_packed_s8(a_ptrs_s8_hi, wei_ptr, dst_ptr1, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                            } else {
                                ker_block4x4_mmla_packed_u8(a_ptrs_u8_lo, wei_ptr, dst_ptr0, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                                ker_block4x4_mmla_packed_u8(a_ptrs_u8_hi, wei_ptr, dst_ptr1, kxk_k, 0,
                                                           dst_row_stride_bytes, kernel_accum);
                            }
                            add_comp_bias(dst_ptr0, 4, 4, oc_tail);
                            add_comp_bias(dst_ptr1, 4, 4, oc_tail);
                        }
                    }
                }

                for (; m + 4 <= m_block; m += 4) {
                    const int8_t* a_ptrs_s8[4] = {
                        im2col_s8.data() + (m + 0) * kxk_k,
                        im2col_s8.data() + (m + 1) * kxk_k,
                        im2col_s8.data() + (m + 2) * kxk_k,
                        im2col_s8.data() + (m + 3) * kxk_k,
                    };
                    const uint8_t* a_ptrs_u8[4] = {
                        im2col_u8.data() + (m + 0) * kxk_k,
                        im2col_u8.data() + (m + 1) * kxk_k,
                        im2col_u8.data() + (m + 2) * kxk_k,
                        im2col_u8.data() + (m + 3) * kxk_k,
                    };
                    size_t oc = 0;
                    for (; use_block16 && oc + 16 <= OC; oc += 16) {
                        const int8_t* wei_ptr =
                            packed_wei_brgemm_mmla16_fused_.data() +
                            (oc / 16) * packed_wei_brgemm_mmla16_fused_stride_;
                        int32_t* dst_ptr = dst_base + (m_start + m) * OC + oc;
                        if (use_mmla_s8) {
                            ker_block4x16_mmla_packed_s8(a_ptrs_s8, wei_ptr, dst_ptr, kxk_k, 0,
                                                         dst_row_stride_bytes, kernel_accum);
                        } else {
                            ker_block4x16_mmla_packed_u8(a_ptrs_u8, wei_ptr, dst_ptr, kxk_k, 0,
                                                         dst_row_stride_bytes, kernel_accum);
                        }
                        add_comp_bias(dst_ptr, 4, 16, oc);
                    }
                    for (; use_block8 && oc + 8 <= OC; oc += 8) {
                        const int8_t* wei_ptr =
                            packed_wei_brgemm_mmla8_fused_.data() +
                            (oc / 8) * packed_wei_brgemm_mmla8_fused_stride_;
                        int32_t* dst_ptr = dst_base + (m_start + m) * OC + oc;
                        if (use_mmla_s8) {
                            ker_block4x8_mmla_packed_s8(a_ptrs_s8, wei_ptr, dst_ptr, kxk_k, 0,
                                                        dst_row_stride_bytes, kernel_accum);
                        } else {
                            ker_block4x8_mmla_packed_u8(a_ptrs_u8, wei_ptr, dst_ptr, kxk_k, 0,
                                                        dst_row_stride_bytes, kernel_accum);
                        }
                        add_comp_bias(dst_ptr, 4, 8, oc);
                    }
                    for (; use_block4 && oc + 4 <= OC; oc += 4) {
                        const int8_t* wei_ptr =
                            packed_wei_brgemm_mmla4_fused_.data() +
                            (oc / 4) * packed_wei_brgemm_mmla4_fused_stride_;
                        int32_t* dst_ptr = dst_base + (m_start + m) * OC + oc;
                        if (use_mmla_s8) {
                            ker_block4x4_mmla_packed_s8(a_ptrs_s8, wei_ptr, dst_ptr, kxk_k, 0,
                                                        dst_row_stride_bytes, kernel_accum);
                        } else {
                            ker_block4x4_mmla_packed_u8(a_ptrs_u8, wei_ptr, dst_ptr, kxk_k, 0,
                                                        dst_row_stride_bytes, kernel_accum);
                        }
                        add_comp_bias(dst_ptr, 4, 4, oc);
                    }
                }

                for (; m < m_block; ++m) {
                    const uint8_t* a_ptr_u8 = im2col_u8.data() + m * kxk_k;
                    const int8_t* a_ptr_s8 = im2col_s8.data() + m * kxk_k;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        const int8_t* wei_ptr = wei.ptr<int8_t>(oc, 0, 0, 0);
                        int32_t acc = 0;
                        size_t k = 0;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const int8_t* w_ptr = wei_ptr + (kh * KW + kw) * IC;
                                if (src_signed) {
                                    const int8_t* a_ptr = a_ptr_s8 + k;
                                    for (size_t ic = 0; ic < IC; ++ic) {
                                        acc += static_cast<int32_t>(a_ptr[ic]) * static_cast<int32_t>(w_ptr[ic]);
                                    }
                                } else {
                                    const uint8_t* a_ptr = a_ptr_u8 + k;
                                    for (size_t ic = 0; ic < IC; ++ic) {
                                        acc += static_cast<int32_t>(a_ptr[ic]) * static_cast<int32_t>(w_ptr[ic]);
                                    }
                                }
                                k += IC;
                            }
                        }
                        if (use_row_prefill) {
                            acc += dst_base[(m_start + m) * OC + oc];
                        } else {
                            if (has_comp_u8) {
                                acc += wei_comp_brgemm_[oc];
                            }
                            if (has_bias) {
                                acc += bias_ptr[oc];
                            }
                        }
                        dst_base[(m_start + m) * OC + oc] = acc;
                    }
                }
            });
            return;
        }

        if (use_brgemm) {
            auto& brgemm_kernel = src_signed ? brgemm_1x1_s8_ : brgemm_1x1_u8_;
            if (!brgemm_kernel || brgemm_1x1_oc_ != OC || brgemm_1x1_ic_ != IC || brgemm_1x1_lda_ != lda_brgemm ||
                brgemm_1x1_ldc_ != ldc_brgemm) {
                brgemm_kernel =
                    std::make_shared<BrgemmInt8Kernel>(brgemm_1x1_m_blk_, OC, IC, lda_brgemm, OC, ldc_brgemm, src_signed);
                brgemm_1x1_oc_ = OC;
                brgemm_1x1_ic_ = IC;
                brgemm_1x1_lda_ = lda_brgemm;
                brgemm_1x1_ldc_ = ldc_brgemm;
            }
        }

        const size_t pack_m = (use_kxk_mmla_fused && OW >= 8) ? 8 : 4;
        const bool prefer_8x8_fused =
            use_packed8_kxk_mmla_fused && (pack_m == 8) && !(use_packed16_kxk_mmla_fused && (OC % 16 == 0));
        const size_t ow_step = use_kxk_mmla_fused ? pack_m : brgemm_1x1_m_blk_;
        const size_t ow_blocks = (OW + ow_step - 1) / ow_step;
        parallel_for3d(N, OH, ow_blocks, [&](size_t n, size_t oh, size_t owb) {
            static thread_local std::vector<uint8_t> packed_src_u8_tls;
            static thread_local std::vector<int8_t> packed_src_s8_tls;
            if (use_kxk_mmla_fused) {
                const size_t pack_elems = pack_m * kxk_kp;
                if (src_signed) {
                    packed_src_s8_tls.resize(pack_elems);
                } else {
                    packed_src_u8_tls.resize(pack_elems);
                }
            } else {
                packed_src_u8_tls.clear();
                packed_src_s8_tls.clear();
            }

            const uint8_t* src_n = src.ptr<uint8_t>(n, 0, 0, 0);
            const ptrdiff_t ih_base = static_cast<ptrdiff_t>(oh * conv_stride_h) - pad_t;
            const bool full_h = ih_base >= 0 && (ih_base + static_cast<ptrdiff_t>(KH)) <= static_cast<ptrdiff_t>(IH);
            size_t ow = owb * ow_step;
            if (ow >= OW) {
                return;
            }
            const size_t ow_end = std::min(OW, ow + ow_step);
            while (ow < ow_end) {
                    const ptrdiff_t iw_base = static_cast<ptrdiff_t>(ow * conv_stride_w) - pad_l;
                    const bool full_w =
                        (use_dot_s8 || use_dot_u8 || use_packed4_kxk_mmla || use_packed8_kxk_mmla ||
                         use_packed16_kxk_mmla) &&
                        full_h && (ow + 4 <= ow_end) && iw_base >= 0 &&
                        (iw_base + static_cast<ptrdiff_t>(KW - 1 + 3 * conv_stride_w)) <
                            static_cast<ptrdiff_t>(IW);
                        const bool can_fused4 = use_kxk_mmla_fused && (ow + 4 <= ow_end);
                        const bool can_fused8 = use_kxk_mmla_fused && (pack_m == 8) && (ow + 8 <= ow_end);
                    const bool full_w_fused4 =
                        can_fused4 && full_h && iw_base >= 0 &&
                        (iw_base + static_cast<ptrdiff_t>(KW - 1 + 3 * conv_stride_w)) <
                            static_cast<ptrdiff_t>(IW);
                        const bool full_w_fused8 =
                            can_fused8 && full_h && iw_base >= 0 &&
                            (iw_base + static_cast<ptrdiff_t>(KW - 1 + 7 * conv_stride_w)) <
                                static_cast<ptrdiff_t>(IW);
                        if (can_fused4) {
                            uint8_t* packed_src_u8 = packed_src_u8_tls.empty() ? nullptr : packed_src_u8_tls.data();
                            int8_t* packed_src_s8 = packed_src_s8_tls.empty() ? nullptr : packed_src_s8_tls.data();
                            const size_t ic_blocks = IC / 8;
                            // Avoid running the "partial-pack" slow path on 8-wide blocks (common for pad>0):
                            // if the 8-wide window is not fully inside bounds, fall back to 4-wide.
                            const size_t m_block = (can_fused8 && full_w_fused8) ? 8 : 4;
                            const size_t pair_count = m_block / 2;
                            const bool full_pack = (m_block == 8) ? full_w_fused8 : full_w_fused4;
                            static thread_local std::vector<uint8_t> zero_row_tls;
                            if (!full_pack) {
                                if (zero_row_tls.size() != IC) {
                                zero_row_tls.assign(IC, 0);
                            }
                        }

                        if (src_signed) {
                            int8_t* pair_ptrs[4] = {packed_src_s8,
                                                    packed_src_s8 + 2 * kxk_kp,
                                                    packed_src_s8 + 4 * kxk_kp,
                                                    packed_src_s8 + 6 * kxk_kp};
                            if (kxk_kp != kxk_k) {
                                for (size_t p = 0; p < pair_count; ++p) {
                                    std::memset(pair_ptrs[p], 0, 2 * kxk_kp);
                                }
                            }
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh);
                                if (full_pack) {
                                    const size_t src_h_off = static_cast<size_t>(ih) * stride_h;
                                    const uint8_t* rows[8] = {};
                                    for (size_t m = 0; m < m_block; ++m) {
                                        rows[m] = src_n + src_h_off +
                                                  static_cast<size_t>(iw_base +
                                                                      static_cast<ptrdiff_t>(m * conv_stride_w)) *
                                                      stride_w;
                                    }
                                    for (size_t kw = 0; kw < KW; ++kw) {
                                        const size_t block_base = (kh * KW + kw) * ic_blocks;
                                        const uint8_t* src0 = rows[0] + kw * stride_w;
                                        const uint8_t* src1 = rows[1] + kw * stride_w;
                                        const uint8_t* src2 = rows[2] + kw * stride_w;
                                        const uint8_t* src3 = rows[3] + kw * stride_w;
                                        uint8_t* dst0 = reinterpret_cast<uint8_t*>(pair_ptrs[0]) + block_base * 16;
                                        uint8_t* dst1 = reinterpret_cast<uint8_t*>(pair_ptrs[1]) + block_base * 16;
                                        if (m_block == 8) {
                                            const uint8_t* src4 = rows[4] + kw * stride_w;
                                            const uint8_t* src5 = rows[5] + kw * stride_w;
                                            const uint8_t* src6 = rows[6] + kw * stride_w;
                                            const uint8_t* src7 = rows[7] + kw * stride_w;
                                            uint8_t* dst2 =
                                                reinterpret_cast<uint8_t*>(pair_ptrs[2]) + block_base * 16;
                                            uint8_t* dst3 =
                                                reinterpret_cast<uint8_t*>(pair_ptrs[3]) + block_base * 16;
                                            pack_pairs8x8(src0, src1, src2, src3, src4, src5, src6, src7,
                                                          dst0, dst1, dst2, dst3, ic_blocks);
                                        } else {
                                            pack_pairs4x8(src0, src1, src2, src3, dst0, dst1, ic_blocks);
                                        }
                                    }
                                } else {
	                                    const size_t kh_off = kh * KW * ic_blocks * 16;
	                                    const size_t kh_bytes = KW * ic_blocks * 16;
	                                    if (ih < 0 || ih >= static_cast<ptrdiff_t>(IH)) {
	                                        for (size_t p = 0; p < pair_count; ++p) {
	                                            std::memset(reinterpret_cast<uint8_t*>(pair_ptrs[p]) + kh_off, 0, kh_bytes);
	                                        }
	                                        continue;
	                                    }

	                                    const uint8_t* const zero_row = zero_row_tls.data();
	                                    const size_t src_h_off = static_cast<size_t>(ih) * stride_h;
	                                    const uint8_t* const row_base = src_n + src_h_off;
	                                    for (size_t p = 0; p < pair_count; ++p) {
	                                        uint8_t* dst = reinterpret_cast<uint8_t*>(pair_ptrs[p]) + kh_off;
	                                        const ptrdiff_t iw0 =
	                                            iw_base + static_cast<ptrdiff_t>((2 * p) * conv_stride_w);
	                                        const ptrdiff_t iw1 = iw0 + static_cast<ptrdiff_t>(conv_stride_w);
	                                        const bool full0 = iw0 >= 0 &&
	                                                           (iw0 + static_cast<ptrdiff_t>(KW - 1)) <
	                                                               static_cast<ptrdiff_t>(IW);
	                                        const bool full1 = iw1 >= 0 &&
	                                                           (iw1 + static_cast<ptrdiff_t>(KW - 1)) <
	                                                               static_cast<ptrdiff_t>(IW);
                                        if (full0 && full1) {
                                            const uint8_t* src0 =
                                                row_base + static_cast<size_t>(iw0) * stride_w;
                                            const uint8_t* src1 =
                                                row_base + static_cast<size_t>(iw1) * stride_w;
                                            for (size_t kw = 0; kw < KW; ++kw) {
                                                pack_pair_8bytes(src0 + kw * stride_w,
                                                                 src1 + kw * stride_w,
                                                                 dst,
                                                                 ic_blocks);
                                                dst += ic_blocks * 16;
                                            }
                                            continue;
                                        }

	                                        // Partial pair: dilation is rejected by this executor, so step is 1.
	                                        const ptrdiff_t kw0_beg = std::max<ptrdiff_t>(0, -iw0);
	                                        const ptrdiff_t kw0_end =
	                                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW),
	                                                                static_cast<ptrdiff_t>(IW) - iw0);
	                                        const ptrdiff_t kw1_beg = std::max<ptrdiff_t>(0, -iw1);
	                                        const ptrdiff_t kw1_end =
	                                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW),
	                                                                static_cast<ptrdiff_t>(IW) - iw1);
	                                        const ptrdiff_t mid_beg = std::max(kw0_beg, kw1_beg);
	                                        const ptrdiff_t mid_end = std::min(kw0_end, kw1_end);

	                                        ptrdiff_t kw = 0;
	                                        for (; kw < mid_beg; ++kw) {
	                                            const ptrdiff_t iw0_kw = iw0 + kw;
	                                            const ptrdiff_t iw1_kw = iw1 + kw;
	                                            const uint8_t* src0 = zero_row;
	                                            const uint8_t* src1 = zero_row;
	                                            if (iw0_kw >= 0 && iw0_kw < static_cast<ptrdiff_t>(IW)) {
	                                                src0 = row_base + static_cast<size_t>(iw0_kw) * stride_w;
	                                            }
	                                            if (iw1_kw >= 0 && iw1_kw < static_cast<ptrdiff_t>(IW)) {
	                                                src1 = row_base + static_cast<size_t>(iw1_kw) * stride_w;
	                                            }
	                                            pack_pair_8bytes(src0, src1, dst, ic_blocks);
	                                            dst += ic_blocks * 16;
	                                        }
                                        if (mid_end > mid_beg) {
                                            const uint8_t* src0 =
                                                row_base + static_cast<size_t>(iw0 + mid_beg) * stride_w;
                                            const uint8_t* src1 =
                                                row_base + static_cast<size_t>(iw1 + mid_beg) * stride_w;
                                            for (ptrdiff_t k = mid_beg; k < mid_end; ++k) {
                                                pack_pair_8bytes(src0, src1, dst, ic_blocks);
                                                dst += ic_blocks * 16;
                                                src0 += stride_w;
                                                src1 += stride_w;
                                            }
                                            kw = mid_end;
                                        }
	                                        for (; kw < static_cast<ptrdiff_t>(KW); ++kw) {
	                                            const ptrdiff_t iw0_kw = iw0 + kw;
	                                            const ptrdiff_t iw1_kw = iw1 + kw;
	                                            const uint8_t* src0 = zero_row;
	                                            const uint8_t* src1 = zero_row;
	                                            if (iw0_kw >= 0 && iw0_kw < static_cast<ptrdiff_t>(IW)) {
	                                                src0 = row_base + static_cast<size_t>(iw0_kw) * stride_w;
	                                            }
	                                            if (iw1_kw >= 0 && iw1_kw < static_cast<ptrdiff_t>(IW)) {
	                                                src1 = row_base + static_cast<size_t>(iw1_kw) * stride_w;
	                                            }
	                                            pack_pair_8bytes(src0, src1, dst, ic_blocks);
	                                            dst += ic_blocks * 16;
	                                        }
	                                    }
	                                }
	                            }
	                        } else {
	                            uint8_t* pair_ptrs[4] = {packed_src_u8,
	                                                     packed_src_u8 + 2 * kxk_kp,
                                                     packed_src_u8 + 4 * kxk_kp,
                                                     packed_src_u8 + 6 * kxk_kp};
                            if (kxk_kp != kxk_k) {
                                for (size_t p = 0; p < pair_count; ++p) {
                                    std::memset(pair_ptrs[p], 0, 2 * kxk_kp);
                                }
                            }
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh);
                                if (full_pack) {
                                    const size_t src_h_off = static_cast<size_t>(ih) * stride_h;
                                    const uint8_t* rows[8] = {};
                                    for (size_t m = 0; m < m_block; ++m) {
                                        rows[m] = src_n + src_h_off +
                                                  static_cast<size_t>(iw_base +
                                                                      static_cast<ptrdiff_t>(m * conv_stride_w)) *
                                                      stride_w;
                                    }
                                    for (size_t kw = 0; kw < KW; ++kw) {
                                        const size_t block_base = (kh * KW + kw) * ic_blocks;
                                        const uint8_t* src0 = rows[0] + kw * stride_w;
                                        const uint8_t* src1 = rows[1] + kw * stride_w;
                                        const uint8_t* src2 = rows[2] + kw * stride_w;
                                        const uint8_t* src3 = rows[3] + kw * stride_w;
                                        uint8_t* dst0 = pair_ptrs[0] + block_base * 16;
                                        uint8_t* dst1 = pair_ptrs[1] + block_base * 16;
                                        if (m_block == 8) {
                                            const uint8_t* src4 = rows[4] + kw * stride_w;
                                            const uint8_t* src5 = rows[5] + kw * stride_w;
                                            const uint8_t* src6 = rows[6] + kw * stride_w;
                                            const uint8_t* src7 = rows[7] + kw * stride_w;
                                            uint8_t* dst2 = pair_ptrs[2] + block_base * 16;
                                            uint8_t* dst3 = pair_ptrs[3] + block_base * 16;
                                            pack_pairs8x8(src0, src1, src2, src3, src4, src5, src6, src7,
                                                          dst0, dst1, dst2, dst3, ic_blocks);
                                        } else {
                                            pack_pairs4x8(src0, src1, src2, src3, dst0, dst1, ic_blocks);
                                        }
                                    }
                                } else {
	                                    const size_t kh_off = kh * KW * ic_blocks * 16;
	                                    const size_t kh_bytes = KW * ic_blocks * 16;
	                                    if (ih < 0 || ih >= static_cast<ptrdiff_t>(IH)) {
	                                        for (size_t p = 0; p < pair_count; ++p) {
	                                            std::memset(pair_ptrs[p] + kh_off, 0, kh_bytes);
	                                        }
	                                        continue;
	                                    }

	                                    const uint8_t* const zero_row = zero_row_tls.data();
	                                    const size_t src_h_off = static_cast<size_t>(ih) * stride_h;
	                                    const uint8_t* const row_base = src_n + src_h_off;
	                                    for (size_t p = 0; p < pair_count; ++p) {
	                                        uint8_t* dst = pair_ptrs[p] + kh_off;
	                                        const ptrdiff_t iw0 =
	                                            iw_base + static_cast<ptrdiff_t>((2 * p) * conv_stride_w);
	                                        const ptrdiff_t iw1 = iw0 + static_cast<ptrdiff_t>(conv_stride_w);
	                                        const bool full0 = iw0 >= 0 &&
	                                                           (iw0 + static_cast<ptrdiff_t>(KW - 1)) <
	                                                               static_cast<ptrdiff_t>(IW);
	                                        const bool full1 = iw1 >= 0 &&
	                                                           (iw1 + static_cast<ptrdiff_t>(KW - 1)) <
	                                                               static_cast<ptrdiff_t>(IW);
                                        if (full0 && full1) {
                                            const uint8_t* src0 =
                                                row_base + static_cast<size_t>(iw0) * stride_w;
                                            const uint8_t* src1 =
                                                row_base + static_cast<size_t>(iw1) * stride_w;
                                            for (size_t kw = 0; kw < KW; ++kw) {
                                                pack_pair_8bytes(src0 + kw * stride_w,
                                                                 src1 + kw * stride_w,
                                                                 dst,
                                                                 ic_blocks);
                                                dst += ic_blocks * 16;
                                            }
                                            continue;
                                        }

	                                        const ptrdiff_t kw0_beg = std::max<ptrdiff_t>(0, -iw0);
	                                        const ptrdiff_t kw0_end =
	                                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW),
	                                                                static_cast<ptrdiff_t>(IW) - iw0);
	                                        const ptrdiff_t kw1_beg = std::max<ptrdiff_t>(0, -iw1);
	                                        const ptrdiff_t kw1_end =
	                                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW),
	                                                                static_cast<ptrdiff_t>(IW) - iw1);
	                                        const ptrdiff_t mid_beg = std::max(kw0_beg, kw1_beg);
	                                        const ptrdiff_t mid_end = std::min(kw0_end, kw1_end);

	                                        ptrdiff_t kw = 0;
	                                        for (; kw < mid_beg; ++kw) {
	                                            const ptrdiff_t iw0_kw = iw0 + kw;
	                                            const ptrdiff_t iw1_kw = iw1 + kw;
	                                            const uint8_t* src0 = zero_row;
	                                            const uint8_t* src1 = zero_row;
	                                            if (iw0_kw >= 0 && iw0_kw < static_cast<ptrdiff_t>(IW)) {
	                                                src0 = row_base + static_cast<size_t>(iw0_kw) * stride_w;
	                                            }
	                                            if (iw1_kw >= 0 && iw1_kw < static_cast<ptrdiff_t>(IW)) {
	                                                src1 = row_base + static_cast<size_t>(iw1_kw) * stride_w;
	                                            }
	                                            pack_pair_8bytes(src0, src1, dst, ic_blocks);
	                                            dst += ic_blocks * 16;
	                                        }
                                        if (mid_end > mid_beg) {
                                            const uint8_t* src0 =
                                                row_base + static_cast<size_t>(iw0 + mid_beg) * stride_w;
                                            const uint8_t* src1 =
                                                row_base + static_cast<size_t>(iw1 + mid_beg) * stride_w;
                                            for (ptrdiff_t k = mid_beg; k < mid_end; ++k) {
                                                pack_pair_8bytes(src0, src1, dst, ic_blocks);
                                                dst += ic_blocks * 16;
                                                src0 += stride_w;
                                                src1 += stride_w;
                                            }
                                            kw = mid_end;
                                        }
	                                        for (; kw < static_cast<ptrdiff_t>(KW); ++kw) {
	                                            const ptrdiff_t iw0_kw = iw0 + kw;
	                                            const ptrdiff_t iw1_kw = iw1 + kw;
	                                            const uint8_t* src0 = zero_row;
	                                            const uint8_t* src1 = zero_row;
	                                            if (iw0_kw >= 0 && iw0_kw < static_cast<ptrdiff_t>(IW)) {
	                                                src0 = row_base + static_cast<size_t>(iw0_kw) * stride_w;
	                                            }
	                                            if (iw1_kw >= 0 && iw1_kw < static_cast<ptrdiff_t>(IW)) {
	                                                src1 = row_base + static_cast<size_t>(iw1_kw) * stride_w;
	                                            }
	                                            pack_pair_8bytes(src0, src1, dst, ic_blocks);
	                                            dst += ic_blocks * 16;
	                                        }
	                                    }
	                                }
	                            }
	                        }

                        size_t oc = 0;
                        const bool use_12_block_strategy =
                            use_packed12_kxk_mmla_fused && (m_block == 8) &&
                            ((OC % 12) == 0 || (OC % 12) == 4 || (OC % 12) == 8) &&
                            (!use_packed16_kxk_mmla_fused || (OC % 12) == 8);
                        if (use_12_block_strategy) {
                            for (; oc + 12 <= OC; oc += 12) {
                                const int8_t* wei_ptr =
                                    packed_wei_brgemm_mmla12_fused_.data() +
                                    (oc / 12) * packed_wei_brgemm_mmla12_fused_stride_;
                                if (oc + 12 < OC) {
                                    prefetch_l1(wei_ptr + packed_wei_brgemm_mmla12_fused_stride_);
                                }
                                int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                const int8_t* src_ptrs_s8[4] = {
                                    packed_src_s8 ? packed_src_s8 : nullptr,
                                    packed_src_s8 ? packed_src_s8 + 2 * kxk_kp : nullptr,
                                    packed_src_s8 ? packed_src_s8 + 4 * kxk_kp : nullptr,
                                    packed_src_s8 ? packed_src_s8 + 6 * kxk_kp : nullptr,
                                };
                                const uint8_t* src_ptrs_u8[4] = {
                                    packed_src_u8 ? packed_src_u8 : nullptr,
                                    packed_src_u8 ? packed_src_u8 + 2 * kxk_kp : nullptr,
                                    packed_src_u8 ? packed_src_u8 + 4 * kxk_kp : nullptr,
                                    packed_src_u8 ? packed_src_u8 + 6 * kxk_kp : nullptr,
                                };
                                if (use_mmla_s8) {
                                    ker_block8x12_mmla_packed_s8(src_ptrs_s8,
                                                                wei_ptr,
                                                                dst_ptr,
                                                                kxk_kp,
                                                                0,
                                                                OC * sizeof(int32_t),
                                                                0);
                                } else {
                                    ker_block8x12_mmla_packed_u8(src_ptrs_u8,
                                                                wei_ptr,
                                                                dst_ptr,
                                                                kxk_kp,
                                                                0,
                                                                OC * sizeof(int32_t),
                                                                0);
                                }
                                if (use_mmla_u8 && !wei_comp_brgemm_.empty()) {
                                    for (size_t r = 0; r < 8; ++r) {
                                        int32_t* dst_row = dst_ptr + r * OC;
                                        for (size_t c = 0; c < 12; ++c) {
                                            dst_row[c] += wei_comp_brgemm_[oc + c];
                                        }
                                    }
                                }
                                if (bias_ptr) {
                                    for (size_t r = 0; r < 8; ++r) {
                                        int32_t* dst_row = dst_ptr + r * OC;
                                        for (size_t c = 0; c < 12; ++c) {
                                            dst_row[c] += bias_ptr[oc + c];
                                        }
                                    }
                                }
                            }
                            for (; use_packed8_kxk_mmla_fused && oc + 8 <= OC; oc += 8) {
                                const int8_t* wei_ptr =
                                    packed_wei_brgemm_mmla8_fused_.data() +
                                    (oc / 8) * packed_wei_brgemm_mmla8_fused_stride_;
                                if (oc + 8 < OC) {
                                    prefetch_l1(wei_ptr + packed_wei_brgemm_mmla8_fused_stride_);
                                }
                                if (m_block == 8) {
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    const int8_t* src_ptrs_s8[4] = {
                                        packed_src_s8 ? packed_src_s8 : nullptr,
                                        packed_src_s8 ? packed_src_s8 + 2 * kxk_kp : nullptr,
                                        packed_src_s8 ? packed_src_s8 + 4 * kxk_kp : nullptr,
                                        packed_src_s8 ? packed_src_s8 + 6 * kxk_kp : nullptr,
                                    };
                                    const uint8_t* src_ptrs_u8[4] = {
                                        packed_src_u8 ? packed_src_u8 : nullptr,
                                        packed_src_u8 ? packed_src_u8 + 2 * kxk_kp : nullptr,
                                        packed_src_u8 ? packed_src_u8 + 4 * kxk_kp : nullptr,
                                        packed_src_u8 ? packed_src_u8 + 6 * kxk_kp : nullptr,
                                    };
                                    if (use_mmla_s8) {
                                        ker_block8x8_mmla_packed_s8(src_ptrs_s8,
                                                                   wei_ptr,
                                                                   dst_ptr,
                                                                   kxk_kp,
                                                                   0,
                                                                   OC * sizeof(int32_t),
                                                                   0);
                                    } else {
                                        ker_block8x8_mmla_packed_u8(src_ptrs_u8,
                                                                   wei_ptr,
                                                                   dst_ptr,
                                                                   kxk_kp,
                                                                   0,
                                                                   OC * sizeof(int32_t),
                                                                   0);
                                    }
                                    if (use_mmla_u8 && !wei_comp_brgemm_.empty()) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 8; ++c) {
                                                dst_row[c] += wei_comp_brgemm_[oc + c];
                                            }
                                        }
                                    }
                                    if (bias_ptr) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 8; ++c) {
                                                dst_row[c] += bias_ptr[oc + c];
                                            }
                                        }
                                    }
                                } else {
                                    for (size_t blk = 0; blk < m_block; blk += 4) {
                                        int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow + blk);
                                        const size_t pair0_off = (blk / 4) * 4 * kxk_kp;
                                        const int8_t* src_ptrs_s8[4] = {
                                            packed_src_s8 ? packed_src_s8 + pair0_off : nullptr,
                                            packed_src_s8 ? packed_src_s8 + pair0_off + 2 * kxk_kp : nullptr,
                                            nullptr,
                                            nullptr,
                                        };
                                        const uint8_t* src_ptrs_u8[4] = {
                                            packed_src_u8 ? packed_src_u8 + pair0_off : nullptr,
                                            packed_src_u8 ? packed_src_u8 + pair0_off + 2 * kxk_kp : nullptr,
                                            nullptr,
                                            nullptr,
                                        };
                                        if (use_mmla_s8) {
                                            ker_block4x8_mmla_packed_s8_interleaved(src_ptrs_s8,
                                                                                    wei_ptr,
                                                                                    dst_ptr,
                                                                                    kxk_kp,
                                                                                    0,
                                                                                    OC * sizeof(int32_t),
                                                                                    0);
                                        } else {
                                            ker_block4x8_mmla_packed_u8_interleaved(src_ptrs_u8,
                                                                                    wei_ptr,
                                                                                    dst_ptr,
                                                                                    kxk_kp,
                                                                                    0,
                                                                                    OC * sizeof(int32_t),
                                                                                    0);
                                        }
                                        if (use_mmla_u8 && !wei_comp_brgemm_.empty()) {
                                            for (size_t r = 0; r < 4; ++r) {
                                                int32_t* dst_row = dst_ptr + r * OC;
                                                for (size_t c = 0; c < 8; ++c) {
                                                    dst_row[c] += wei_comp_brgemm_[oc + c];
                                                }
                                            }
                                        }
                                        if (bias_ptr) {
                                            for (size_t r = 0; r < 4; ++r) {
                                                int32_t* dst_row = dst_ptr + r * OC;
                                                for (size_t c = 0; c < 8; ++c) {
                                                    dst_row[c] += bias_ptr[oc + c];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            for (; !prefer_8x8_fused && use_packed16_kxk_mmla_fused && oc + 16 <= OC; oc += 16) {
                                const int8_t* wei_ptr =
                                    packed_wei_brgemm_mmla16_fused_.data() +
                                    (oc / 16) * packed_wei_brgemm_mmla16_fused_stride_;
                                if (oc + 16 < OC) {
                                    prefetch_l1(wei_ptr + packed_wei_brgemm_mmla16_fused_stride_);
                                }
                                for (size_t blk = 0; blk < m_block; blk += 4) {
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow + blk);
                                    const size_t pair0_off = (blk / 4) * 4 * kxk_kp;
                                    const int8_t* src_ptrs_s8[4] = {
                                        packed_src_s8 ? packed_src_s8 + pair0_off : nullptr,
                                        packed_src_s8 ? packed_src_s8 + pair0_off + 2 * kxk_kp : nullptr,
                                        nullptr,
                                        nullptr,
                                    };
                                    const uint8_t* src_ptrs_u8[4] = {
                                        packed_src_u8 ? packed_src_u8 + pair0_off : nullptr,
                                        packed_src_u8 ? packed_src_u8 + pair0_off + 2 * kxk_kp : nullptr,
                                        nullptr,
                                        nullptr,
                                    };
                                    if (use_mmla_s8) {
                                        ker_block4x16_mmla_packed_s8_interleaved(src_ptrs_s8,
                                                                                 wei_ptr,
                                                                                 dst_ptr,
                                                                                 kxk_kp,
                                                                                 0,
                                                                                 OC * sizeof(int32_t),
                                                                                 0);
                                    } else {
                                        ker_block4x16_mmla_packed_u8_interleaved(src_ptrs_u8,
                                                                                 wei_ptr,
                                                                                 dst_ptr,
                                                                                 kxk_kp,
                                                                                 0,
                                                                                 OC * sizeof(int32_t),
                                                                                 0);
                                    }
                                    if (use_mmla_u8 && !wei_comp_brgemm_.empty()) {
                                        for (size_t r = 0; r < 4; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 16; ++c) {
                                                dst_row[c] += wei_comp_brgemm_[oc + c];
                                            }
                                        }
                                    }
                                    if (bias_ptr) {
                                        for (size_t r = 0; r < 4; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 16; ++c) {
                                                dst_row[c] += bias_ptr[oc + c];
                                            }
                                        }
                                    }
                                }
                            }
                            for (; use_packed8_kxk_mmla_fused && oc + 8 <= OC; oc += 8) {
                                const int8_t* wei_ptr =
                                    packed_wei_brgemm_mmla8_fused_.data() +
                                    (oc / 8) * packed_wei_brgemm_mmla8_fused_stride_;
                                if (oc + 8 < OC) {
                                    prefetch_l1(wei_ptr + packed_wei_brgemm_mmla8_fused_stride_);
                                }
                                if (m_block == 8) {
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                                    const int8_t* src_ptrs_s8[4] = {
                                        packed_src_s8 ? packed_src_s8 : nullptr,
                                        packed_src_s8 ? packed_src_s8 + 2 * kxk_kp : nullptr,
                                        packed_src_s8 ? packed_src_s8 + 4 * kxk_kp : nullptr,
                                        packed_src_s8 ? packed_src_s8 + 6 * kxk_kp : nullptr,
                                    };
                                    const uint8_t* src_ptrs_u8[4] = {
                                        packed_src_u8 ? packed_src_u8 : nullptr,
                                        packed_src_u8 ? packed_src_u8 + 2 * kxk_kp : nullptr,
                                        packed_src_u8 ? packed_src_u8 + 4 * kxk_kp : nullptr,
                                        packed_src_u8 ? packed_src_u8 + 6 * kxk_kp : nullptr,
                                    };
                                    if (use_mmla_s8) {
                                        ker_block8x8_mmla_packed_s8(src_ptrs_s8,
                                                                   wei_ptr,
                                                                   dst_ptr,
                                                                   kxk_kp,
                                                                   0,
                                                                   OC * sizeof(int32_t),
                                                                   0);
                                    } else {
                                        ker_block8x8_mmla_packed_u8(src_ptrs_u8,
                                                                   wei_ptr,
                                                                   dst_ptr,
                                                                   kxk_kp,
                                                                   0,
                                                                   OC * sizeof(int32_t),
                                                                   0);
                                    }
                                    if (use_mmla_u8 && !wei_comp_brgemm_.empty()) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 8; ++c) {
                                                dst_row[c] += wei_comp_brgemm_[oc + c];
                                            }
                                        }
                                    }
                                    if (bias_ptr) {
                                        for (size_t r = 0; r < 8; ++r) {
                                            int32_t* dst_row = dst_ptr + r * OC;
                                            for (size_t c = 0; c < 8; ++c) {
                                                dst_row[c] += bias_ptr[oc + c];
                                            }
                                        }
                                    }
                                } else {
                                    for (size_t blk = 0; blk < m_block; blk += 4) {
                                        int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow + blk);
                                        const size_t pair0_off = (blk / 4) * 4 * kxk_kp;
                                        const int8_t* src_ptrs_s8[4] = {
                                            packed_src_s8 ? packed_src_s8 + pair0_off : nullptr,
                                            packed_src_s8 ? packed_src_s8 + pair0_off + 2 * kxk_kp : nullptr,
                                            nullptr,
                                            nullptr,
                                        };
                                        const uint8_t* src_ptrs_u8[4] = {
                                            packed_src_u8 ? packed_src_u8 + pair0_off : nullptr,
                                            packed_src_u8 ? packed_src_u8 + pair0_off + 2 * kxk_kp : nullptr,
                                            nullptr,
                                            nullptr,
                                        };
                                        if (use_mmla_s8) {
                                            ker_block4x8_mmla_packed_s8_interleaved(src_ptrs_s8,
                                                                                    wei_ptr,
                                                                                    dst_ptr,
                                                                                    kxk_kp,
                                                                                    0,
                                                                                    OC * sizeof(int32_t),
                                                                                    0);
                                        } else {
                                            ker_block4x8_mmla_packed_u8_interleaved(src_ptrs_u8,
                                                                                    wei_ptr,
                                                                                    dst_ptr,
                                                                                    kxk_kp,
                                                                                    0,
                                                                                    OC * sizeof(int32_t),
                                                                                    0);
                                        }
                                        if (use_mmla_u8 && !wei_comp_brgemm_.empty()) {
                                            for (size_t r = 0; r < 4; ++r) {
                                                int32_t* dst_row = dst_ptr + r * OC;
                                                for (size_t c = 0; c < 8; ++c) {
                                                    dst_row[c] += wei_comp_brgemm_[oc + c];
                                                }
                                            }
                                        }
                                        if (bias_ptr) {
                                            for (size_t r = 0; r < 4; ++r) {
                                                int32_t* dst_row = dst_ptr + r * OC;
                                                for (size_t c = 0; c < 8; ++c) {
                                                    dst_row[c] += bias_ptr[oc + c];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for (; use_packed4_kxk_mmla_fused && oc + 4 <= OC; oc += 4) {
                            const int8_t* wei_ptr =
                                packed_wei_brgemm_mmla4_fused_.data() +
                                (oc / 4) * packed_wei_brgemm_mmla4_fused_stride_;
                            if (oc + 4 < OC) {
                                prefetch_l1(wei_ptr + packed_wei_brgemm_mmla4_fused_stride_);
                            }
                            for (size_t blk = 0; blk < m_block; blk += 4) {
                                int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow + blk);
                                const size_t pair0_off = (blk / 4) * 4 * kxk_kp;
                                const int8_t* src_ptrs_s8[4] = {
                                    packed_src_s8 ? packed_src_s8 + pair0_off : nullptr,
                                    packed_src_s8 ? packed_src_s8 + pair0_off + 2 * kxk_kp : nullptr,
                                    nullptr,
                                    nullptr,
                                };
                                const uint8_t* src_ptrs_u8[4] = {
                                    packed_src_u8 ? packed_src_u8 + pair0_off : nullptr,
                                    packed_src_u8 ? packed_src_u8 + pair0_off + 2 * kxk_kp : nullptr,
                                    nullptr,
                                    nullptr,
                                };
                                if (use_mmla_s8) {
                                    ker_block4x4_mmla_packed_s8_interleaved(src_ptrs_s8,
                                                                            wei_ptr,
                                                                            dst_ptr,
                                                                            kxk_kp,
                                                                            0,
                                                                            OC * sizeof(int32_t),
                                                                            0);
                                } else {
                                    ker_block4x4_mmla_packed_u8_interleaved(src_ptrs_u8,
                                                                            wei_ptr,
                                                                            dst_ptr,
                                                                            kxk_kp,
                                                                            0,
                                                                            OC * sizeof(int32_t),
                                                                            0);
                                }
                                if (use_mmla_u8 && !wei_comp_brgemm_.empty()) {
                                    for (size_t r = 0; r < 4; ++r) {
                                        int32_t* dst_row = dst_ptr + r * OC;
                                        for (size_t c = 0; c < 4; ++c) {
                                            dst_row[c] += wei_comp_brgemm_[oc + c];
                                        }
                                    }
                                }
                                if (bias_ptr) {
                                    for (size_t r = 0; r < 4; ++r) {
                                        int32_t* dst_row = dst_ptr + r * OC;
                                        for (size_t c = 0; c < 4; ++c) {
                                            dst_row[c] += bias_ptr[oc + c];
                                        }
                                    }
                                }
                            }
                        }
                        for (; oc < OC; ++oc) {
                            for (size_t blk = 0; blk < m_block; blk += 4) {
                                for (size_t i = 0; i < 4; ++i) {
                                    int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow + blk + i);
                                    bool wrote = false;
                                    for (size_t kh = 0; kh < KH; ++kh) {
                                        const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh));
                                        const size_t src_h_off = ih * stride_h;
                                        for (size_t kw = 0; kw < KW; ++kw) {
                                            const size_t iw =
                                                static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(kw) +
                                                                    static_cast<ptrdiff_t>((blk + i) * conv_stride_w));
                                            const uint8_t* src_ptr = src_n + src_h_off + iw * stride_w;
                                            const int8_t* wei_ptr =
                                                packed_wei_.data() + ((kh * KW + kw) * OC + oc) * IC;
                                            ker(reinterpret_cast<const uint8_t*>(src_ptr),
                                                wei_ptr,
                                                dst_ptr,
                                                IC,
                                                wrote ? 1 : 0);
                                            wrote = true;
                                        }
                                    }
                                    if (bias_ptr) {
                                        *dst_ptr += bias_ptr[oc];
                                    }
                                }
                            }
                        }
                        ow += m_block;
                        continue;
                    }
                    if (full_w) {
                        size_t oc = 0;
                        for (; use_packed16_kxk_mmla && oc + 16 <= OC; oc += 16) {
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh));
                                const size_t src_h_off = ih * stride_h;
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const size_t khkw = kh * KW + kw;
                                    const size_t iw = static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(kw));
                                    const size_t src_w_off0 = (iw + 0 * conv_stride_w) * stride_w;
                                    const size_t src_w_off1 = (iw + 1 * conv_stride_w) * stride_w;
                                    const size_t src_w_off2 = (iw + 2 * conv_stride_w) * stride_w;
                                    const size_t src_w_off3 = (iw + 3 * conv_stride_w) * stride_w;
                                    const uint8_t* src_ptrs_u8[4] = {
                                        src_n + src_h_off + src_w_off0,
                                        src_n + src_h_off + src_w_off1,
                                        src_n + src_h_off + src_w_off2,
                                        src_n + src_h_off + src_w_off3,
                                    };
                                    const int8_t* wei_ptr =
                                        packed_wei_brgemm_mmla16_.data() +
                                        (khkw * oc_blocks16_mmla + (oc / 16)) * packed_wei_brgemm_mmla16_stride_;
                                    const size_t accum = (kh == 0 && kw == 0) ? 0 : 1;
                                    if (use_mmla_s8) {
                                        const int8_t* src_ptrs_s8[4] = {
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                        };
                                        ker_block4x16_mmla_packed_s8(src_ptrs_s8,
                                                                     wei_ptr,
                                                                     dst_ptr,
                                                                     IC,
                                                                     0,
                                                                     OC * sizeof(int32_t),
                                                                     accum);
                                    } else {
                                        ker_block4x16_mmla_packed_u8(src_ptrs_u8,
                                                                     wei_ptr,
                                                                     dst_ptr,
                                                                     IC,
                                                                     0,
                                                                     OC * sizeof(int32_t),
                                                                     accum);
                                    }
                                }
                            }
                            if (use_mmla_u8 && !wei_comp_brgemm_.empty()) {
                                for (size_t r = 0; r < 4; ++r) {
                                    int32_t* dst_row = dst_ptr + r * OC;
                                    for (size_t c = 0; c < 16; ++c) {
                                        dst_row[c] += wei_comp_brgemm_[oc + c];
                                    }
                                }
                            }
                            if (bias_ptr) {
                                int32_t* dst_row0 = dst_ptr;
                                int32_t* dst_row1 = dst_row0 + OC;
                                int32_t* dst_row2 = dst_row1 + OC;
                                int32_t* dst_row3 = dst_row2 + OC;
                                const int32_t b0 = bias_ptr[oc];
                                const int32_t b1 = bias_ptr[oc + 1];
                                const int32_t b2 = bias_ptr[oc + 2];
                                const int32_t b3 = bias_ptr[oc + 3];
                                const int32_t b4 = bias_ptr[oc + 4];
                                const int32_t b5 = bias_ptr[oc + 5];
                                const int32_t b6 = bias_ptr[oc + 6];
                                const int32_t b7 = bias_ptr[oc + 7];
                                const int32_t b8 = bias_ptr[oc + 8];
                                const int32_t b9 = bias_ptr[oc + 9];
                                const int32_t b10 = bias_ptr[oc + 10];
                                const int32_t b11 = bias_ptr[oc + 11];
                                const int32_t b12 = bias_ptr[oc + 12];
                                const int32_t b13 = bias_ptr[oc + 13];
                                const int32_t b14 = bias_ptr[oc + 14];
                                const int32_t b15 = bias_ptr[oc + 15];
                                dst_row0[0] += b0;
                                dst_row0[1] += b1;
                                dst_row0[2] += b2;
                                dst_row0[3] += b3;
                                dst_row0[4] += b4;
                                dst_row0[5] += b5;
                                dst_row0[6] += b6;
                                dst_row0[7] += b7;
                                dst_row0[8] += b8;
                                dst_row0[9] += b9;
                                dst_row0[10] += b10;
                                dst_row0[11] += b11;
                                dst_row0[12] += b12;
                                dst_row0[13] += b13;
                                dst_row0[14] += b14;
                                dst_row0[15] += b15;
                                dst_row1[0] += b0;
                                dst_row1[1] += b1;
                                dst_row1[2] += b2;
                                dst_row1[3] += b3;
                                dst_row1[4] += b4;
                                dst_row1[5] += b5;
                                dst_row1[6] += b6;
                                dst_row1[7] += b7;
                                dst_row1[8] += b8;
                                dst_row1[9] += b9;
                                dst_row1[10] += b10;
                                dst_row1[11] += b11;
                                dst_row1[12] += b12;
                                dst_row1[13] += b13;
                                dst_row1[14] += b14;
                                dst_row1[15] += b15;
                                dst_row2[0] += b0;
                                dst_row2[1] += b1;
                                dst_row2[2] += b2;
                                dst_row2[3] += b3;
                                dst_row2[4] += b4;
                                dst_row2[5] += b5;
                                dst_row2[6] += b6;
                                dst_row2[7] += b7;
                                dst_row2[8] += b8;
                                dst_row2[9] += b9;
                                dst_row2[10] += b10;
                                dst_row2[11] += b11;
                                dst_row2[12] += b12;
                                dst_row2[13] += b13;
                                dst_row2[14] += b14;
                                dst_row2[15] += b15;
                                dst_row3[0] += b0;
                                dst_row3[1] += b1;
                                dst_row3[2] += b2;
                                dst_row3[3] += b3;
                                dst_row3[4] += b4;
                                dst_row3[5] += b5;
                                dst_row3[6] += b6;
                                dst_row3[7] += b7;
                                dst_row3[8] += b8;
                                dst_row3[9] += b9;
                                dst_row3[10] += b10;
                                dst_row3[11] += b11;
                                dst_row3[12] += b12;
                                dst_row3[13] += b13;
                                dst_row3[14] += b14;
                                dst_row3[15] += b15;
                            }
                        }
                        for (; use_packed8_kxk_mmla && oc + 8 <= OC; oc += 8) {
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh));
                                const size_t src_h_off = ih * stride_h;
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const size_t khkw = kh * KW + kw;
                                    const size_t iw = static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(kw));
                                    const size_t src_w_off0 = (iw + 0 * conv_stride_w) * stride_w;
                                    const size_t src_w_off1 = (iw + 1 * conv_stride_w) * stride_w;
                                    const size_t src_w_off2 = (iw + 2 * conv_stride_w) * stride_w;
                                    const size_t src_w_off3 = (iw + 3 * conv_stride_w) * stride_w;
                                    const uint8_t* src_ptrs_u8[4] = {
                                        src_n + src_h_off + src_w_off0,
                                        src_n + src_h_off + src_w_off1,
                                        src_n + src_h_off + src_w_off2,
                                        src_n + src_h_off + src_w_off3,
                                    };
                                    const int8_t* wei_ptr =
                                        packed_wei_brgemm_mmla8_.data() +
                                        (khkw * oc_blocks8_mmla + (oc / 8)) * packed_wei_brgemm_mmla8_stride_;
                                    const size_t accum = (kh == 0 && kw == 0) ? 0 : 1;
                                    if (use_mmla_s8) {
                                        const int8_t* src_ptrs_s8[4] = {
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                        };
                                        ker_block4x8_mmla_packed_s8(src_ptrs_s8,
                                                                    wei_ptr,
                                                                    dst_ptr,
                                                                    IC,
                                                                    0,
                                                                    OC * sizeof(int32_t),
                                                                    accum);
                                    } else {
                                        ker_block4x8_mmla_packed_u8(src_ptrs_u8,
                                                                    wei_ptr,
                                                                    dst_ptr,
                                                                    IC,
                                                                    0,
                                                                    OC * sizeof(int32_t),
                                                                    accum);
                                    }
                                }
                            }
                            if (use_mmla_u8 && !wei_comp_brgemm_.empty()) {
                                for (size_t r = 0; r < 4; ++r) {
                                    int32_t* dst_row = dst_ptr + r * OC;
                                    for (size_t c = 0; c < 8; ++c) {
                                        dst_row[c] += wei_comp_brgemm_[oc + c];
                                    }
                                }
                            }
                            if (bias_ptr) {
                                int32_t* dst_row0 = dst_ptr;
                                int32_t* dst_row1 = dst_row0 + OC;
                                int32_t* dst_row2 = dst_row1 + OC;
                                int32_t* dst_row3 = dst_row2 + OC;
                                const int32_t b0 = bias_ptr[oc];
                                const int32_t b1 = bias_ptr[oc + 1];
                                const int32_t b2 = bias_ptr[oc + 2];
                                const int32_t b3 = bias_ptr[oc + 3];
                                const int32_t b4 = bias_ptr[oc + 4];
                                const int32_t b5 = bias_ptr[oc + 5];
                                const int32_t b6 = bias_ptr[oc + 6];
                                const int32_t b7 = bias_ptr[oc + 7];
                                dst_row0[0] += b0;
                                dst_row0[1] += b1;
                                dst_row0[2] += b2;
                                dst_row0[3] += b3;
                                dst_row0[4] += b4;
                                dst_row0[5] += b5;
                                dst_row0[6] += b6;
                                dst_row0[7] += b7;
                                dst_row1[0] += b0;
                                dst_row1[1] += b1;
                                dst_row1[2] += b2;
                                dst_row1[3] += b3;
                                dst_row1[4] += b4;
                                dst_row1[5] += b5;
                                dst_row1[6] += b6;
                                dst_row1[7] += b7;
                                dst_row2[0] += b0;
                                dst_row2[1] += b1;
                                dst_row2[2] += b2;
                                dst_row2[3] += b3;
                                dst_row2[4] += b4;
                                dst_row2[5] += b5;
                                dst_row2[6] += b6;
                                dst_row2[7] += b7;
                                dst_row3[0] += b0;
                                dst_row3[1] += b1;
                                dst_row3[2] += b2;
                                dst_row3[3] += b3;
                                dst_row3[4] += b4;
                                dst_row3[5] += b5;
                                dst_row3[6] += b6;
                                dst_row3[7] += b7;
                            }
                        }
                        for (; oc + 4 <= OC; oc += 4) {
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh));
                                const size_t src_h_off = ih * stride_h;
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const size_t khkw = kh * KW + kw;
                                    const size_t iw = static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(kw));
                                    const size_t src_w_off0 = (iw + 0 * conv_stride_w) * stride_w;
                                    const size_t src_w_off1 = (iw + 1 * conv_stride_w) * stride_w;
                                    const size_t src_w_off2 = (iw + 2 * conv_stride_w) * stride_w;
                                    const size_t src_w_off3 = (iw + 3 * conv_stride_w) * stride_w;
                                    const uint8_t* src_ptrs_u8[4] = {
                                        src_n + src_h_off + src_w_off0,
                                        src_n + src_h_off + src_w_off1,
                                        src_n + src_h_off + src_w_off2,
                                        src_n + src_h_off + src_w_off3,
                                    };
                                    const int8_t* wei_ptr =
                                        use_packed4_kxk_mmla
                                            ? (packed_wei_brgemm_mmla4_.data() +
                                               (khkw * oc_blocks4_mmla + (oc / 4)) *
                                                   packed_wei_brgemm_mmla4_stride_)
                                            : (use_packed4_kxk
                                                   ? (packed_wei_brgemm_dot4_.data() +
                                                      (khkw * oc_blocks4 + (oc / 4)) * packed_wei_brgemm_dot4_stride_)
                                                   : (packed_wei_.data() + (khkw * OC + oc) * IC));
                                    const size_t accum = (kh == 0 && kw == 0) ? 0 : 1;
                                    if (use_packed4_kxk_mmla) {
                                        if (use_mmla_s8) {
                                            const int8_t* src_ptrs_s8[4] = {
                                                reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                                reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                                reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                                reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                            };
                                            ker_block4x4_mmla_packed_s8(src_ptrs_s8,
                                                                        wei_ptr,
                                                                        dst_ptr,
                                                                        IC,
                                                                        0,
                                                                        OC * sizeof(int32_t),
                                                                        accum);
                                        } else {
                                            ker_block4x4_mmla_packed_u8(src_ptrs_u8,
                                                                        wei_ptr,
                                                                        dst_ptr,
                                                                        IC,
                                                                        0,
                                                                        OC * sizeof(int32_t),
                                                                        accum);
                                        }
                                    } else if (use_dot_s8) {
                                        const int8_t* src_ptrs_s8[4] = {
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                            reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                        };
                                        if (use_packed4_kxk) {
                                            ker_block4x4_dot_packed(src_ptrs_s8,
                                                                    wei_ptr,
                                                                    dst_ptr,
                                                                    IC,
                                                                    0,
                                                                    OC * sizeof(int32_t),
                                                                    accum);
                                        } else {
                                            ker_block4x4_dot(src_ptrs_s8,
                                                             wei_ptr,
                                                             dst_ptr,
                                                             IC,
                                                             IC,
                                                             OC * sizeof(int32_t),
                                                             accum);
                                        }
                                    } else {
                                        if (use_packed4_kxk) {
                                            ker_block4x4_udot_packed(src_ptrs_u8,
                                                                     wei_ptr,
                                                                     dst_ptr,
                                                                     IC,
                                                                     0,
                                                                     OC * sizeof(int32_t),
                                                                     accum);
                                        } else {
                                            ker_block4x4_udot(src_ptrs_u8,
                                                              wei_ptr,
                                                              dst_ptr,
                                                              IC,
                                                              IC,
                                                              OC * sizeof(int32_t),
                                                              accum);
                                        }
                                    }
                                }
                            }
                            if ((use_dot_u8 || use_mmla_u8) && !wei_comp_brgemm_.empty()) {
                                int32_t* dst_row0 = dst_ptr;
                                int32_t* dst_row1 = dst_row0 + OC;
                                int32_t* dst_row2 = dst_row1 + OC;
                                int32_t* dst_row3 = dst_row2 + OC;
                                dst_row0[0] += wei_comp_brgemm_[oc];
                                dst_row0[1] += wei_comp_brgemm_[oc + 1];
                                dst_row0[2] += wei_comp_brgemm_[oc + 2];
                                dst_row0[3] += wei_comp_brgemm_[oc + 3];
                                dst_row1[0] += wei_comp_brgemm_[oc];
                                dst_row1[1] += wei_comp_brgemm_[oc + 1];
                                dst_row1[2] += wei_comp_brgemm_[oc + 2];
                                dst_row1[3] += wei_comp_brgemm_[oc + 3];
                                dst_row2[0] += wei_comp_brgemm_[oc];
                                dst_row2[1] += wei_comp_brgemm_[oc + 1];
                                dst_row2[2] += wei_comp_brgemm_[oc + 2];
                                dst_row2[3] += wei_comp_brgemm_[oc + 3];
                                dst_row3[0] += wei_comp_brgemm_[oc];
                                dst_row3[1] += wei_comp_brgemm_[oc + 1];
                                dst_row3[2] += wei_comp_brgemm_[oc + 2];
                                dst_row3[3] += wei_comp_brgemm_[oc + 3];
                            }
                            if (bias_ptr) {
                                int32_t* dst_row0 = dst_ptr;
                                int32_t* dst_row1 = dst_row0 + OC;
                                int32_t* dst_row2 = dst_row1 + OC;
                                int32_t* dst_row3 = dst_row2 + OC;
                                const int32_t b0 = bias_ptr[oc];
                                const int32_t b1 = bias_ptr[oc + 1];
                                const int32_t b2 = bias_ptr[oc + 2];
                                const int32_t b3 = bias_ptr[oc + 3];
                                dst_row0[0] += b0;
                                dst_row0[1] += b1;
                                dst_row0[2] += b2;
                                dst_row0[3] += b3;
                                dst_row1[0] += b0;
                                dst_row1[1] += b1;
                                dst_row1[2] += b2;
                                dst_row1[3] += b3;
                                dst_row2[0] += b0;
                                dst_row2[1] += b1;
                                dst_row2[2] += b2;
                                dst_row2[3] += b3;
                                dst_row3[0] += b0;
                                dst_row3[1] += b1;
                                dst_row3[2] += b2;
                                dst_row3[3] += b3;
                            }
                        }
                        for (; oc < OC; ++oc) {
                            for (size_t i = 0; i < 4; ++i) {
                                int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow + i);
                                bool wrote = false;
                                for (size_t kh = 0; kh < KH; ++kh) {
                                    const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh));
                                    const size_t src_h_off = ih * stride_h;
                                    for (size_t kw = 0; kw < KW; ++kw) {
                                        const size_t iw = static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(kw) +
                                                                              static_cast<ptrdiff_t>(i * conv_stride_w));
                                        const uint8_t* src_ptr = src_n + src_h_off + iw * stride_w;
                                        const int8_t* wei_ptr =
                                            packed_wei_.data() + ((kh * KW + kw) * OC + oc) * IC;
                                        ker(reinterpret_cast<const uint8_t*>(src_ptr),
                                            wei_ptr,
                                            dst_ptr,
                                            IC,
                                            wrote ? 1 : 0);
                                        wrote = true;
                                    }
                                }
                                if (bias_ptr) {
                                    *dst_ptr += bias_ptr[oc];
                                }
                            }
                        }
                        ow += 4;
                        continue;
                    }
                    const bool full_w_brgemm =
                        use_brgemm && full_h && (ow + brgemm_1x1_m_blk_ <= ow_end) && iw_base >= 0 &&
                        (iw_base + static_cast<ptrdiff_t>(KW - 1 + (brgemm_1x1_m_blk_ - 1) * conv_stride_w)) <
                            static_cast<ptrdiff_t>(IW);
                    if (full_w_brgemm) {
                        auto* brgemm_kernel = src_signed ? brgemm_1x1_s8_.get() : brgemm_1x1_u8_.get();
                        const bool use_comp =
                            !src_signed && brgemm_kernel && !brgemm_kernel->uses_brgemm() && !wei_comp_brgemm_.empty();
                        const int32_t* comp_ptr = use_comp ? wei_comp_brgemm_.data() : nullptr;
                        const bool use_onednn_brgemm = brgemm_kernel && brgemm_kernel->uses_brgemm();
                        const int8_t* wei_base =
                            use_onednn_brgemm ? packed_wei_brgemm_.data() : packed_wei_brgemm_col_.data();
                        const size_t wei_stride = use_onednn_brgemm ? (IC * OC) : (OC * IC);
                        brgemm_batch_element_t batch[25];
                        size_t idx = 0;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh));
                            const size_t src_h_off = ih * stride_h;
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const size_t iw = static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(kw));
                                const uint8_t* src_ptr = src_n + src_h_off + iw * stride_w;
                                const int8_t* wei_ptr = wei_base + (kh * KW + kw) * wei_stride;
                                batch[idx].ptr.A = src_ptr;
                                batch[idx].ptr.B = wei_ptr;
                                ++idx;
                            }
                        }
                        int32_t* dst_ptr = dst.ptr<int32_t>(n, 0, oh, ow);
                        brgemm_kernel->execute_batch(batch, static_cast<int>(idx), dst_ptr);
                        if (bias_ptr || use_comp) {
                            for (size_t m = 0; m < brgemm_1x1_m_blk_; ++m) {
                                int32_t* dst_block = dst_ptr + m * ldc_brgemm;
                                for (size_t oc = 0; oc < OC; ++oc) {
                                    const int32_t bias_val = bias_ptr ? bias_ptr[oc] : 0;
                                    const int32_t comp_val = comp_ptr ? comp_ptr[oc] : 0;
                                    dst_block[oc] += bias_val + comp_val;
                                }
                            }
                        }
                        ow += brgemm_1x1_m_blk_;
                        continue;
                    }

                    size_t oc = 0;
                    if (use_dot_s8) {
                        for (; oc + 8 <= OC; oc += 8) {
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            bool wrote = false;
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh);
                                if (ih < 0 || ih >= static_cast<ptrdiff_t>(IH)) {
                                    continue;
                                }
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const ptrdiff_t iw = iw_base + static_cast<ptrdiff_t>(kw);
                                    if (iw < 0 || iw >= static_cast<ptrdiff_t>(IW)) {
                                        continue;
                                    }
                                    const uint8_t* src_ptr =
                                        src_n + static_cast<size_t>(ih) * stride_h + static_cast<size_t>(iw) * stride_w;
                                    const size_t khkw = kh * KW + kw;
                                    const int8_t* wei_ptr =
                                        use_packed8_kxk
                                            ? (packed_wei_brgemm_dot8_.data() +
                                               (khkw * oc_blocks8 + (oc / 8)) * packed_wei_brgemm_dot8_stride_)
                                            : (packed_wei_.data() + (khkw * OC + oc) * IC);
                                    if (use_packed8_kxk) {
                                        ker_block8_dot_packed(reinterpret_cast<const int8_t*>(src_ptr),
                                                              wei_ptr,
                                                              dst_ptr,
                                                              IC,
                                                              0,
                                                              wrote ? 1 : 0);
                                    } else {
                                        ker_block8_dot(reinterpret_cast<const int8_t*>(src_ptr),
                                                       wei_ptr,
                                                       dst_ptr,
                                                       IC,
                                                       IC,
                                                       wrote ? 1 : 0);
                                    }
                                    wrote = true;
                                }
                            }
                            if (!wrote) {
                                for (size_t i = 0; i < 8; ++i) {
                                    dst_ptr[i] = bias_ptr ? bias_ptr[oc + i] : 0;
                                }
                            } else if (bias_ptr) {
                                for (size_t i = 0; i < 8; ++i) {
                                    dst_ptr[i] += bias_ptr[oc + i];
                                }
                            }
                        }
                    } else if (use_dot_u8) {
                        for (; oc + 8 <= OC; oc += 8) {
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            bool wrote = false;
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh);
                                if (ih < 0 || ih >= static_cast<ptrdiff_t>(IH)) {
                                    continue;
                                }
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const ptrdiff_t iw = iw_base + static_cast<ptrdiff_t>(kw);
                                    if (iw < 0 || iw >= static_cast<ptrdiff_t>(IW)) {
                                        continue;
                                    }
                                    const uint8_t* src_ptr =
                                        src_n + static_cast<size_t>(ih) * stride_h + static_cast<size_t>(iw) * stride_w;
                                    const size_t khkw = kh * KW + kw;
                                    const int8_t* wei_ptr =
                                        use_packed8_kxk
                                            ? (packed_wei_brgemm_dot8_.data() +
                                               (khkw * oc_blocks8 + (oc / 8)) * packed_wei_brgemm_dot8_stride_)
                                            : (packed_wei_.data() + (khkw * OC + oc) * IC);
                                    if (use_packed8_kxk) {
                                        ker_block8_udot_packed(src_ptr, wei_ptr, dst_ptr, IC, 0, wrote ? 1 : 0);
                                    } else {
                                        ker_block8_udot(src_ptr, wei_ptr, dst_ptr, IC, IC, wrote ? 1 : 0);
                                    }
                                    wrote = true;
                                }
                            }
                            if (!wrote) {
                                for (size_t i = 0; i < 8; ++i) {
                                    dst_ptr[i] = bias_ptr ? bias_ptr[oc + i] : 0;
                                }
                            } else {
                                if (!wei_comp_brgemm_.empty()) {
                                    for (size_t i = 0; i < 8; ++i) {
                                        dst_ptr[i] += wei_comp_brgemm_[oc + i];
                                    }
                                }
                                if (bias_ptr) {
                                    for (size_t i = 0; i < 8; ++i) {
                                        dst_ptr[i] += bias_ptr[oc + i];
                                    }
                                }
                            }
                        }
                    }
                    for (; oc + 4 <= OC; oc += 4) {
                        int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                        bool wrote = false;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh);
                            if (ih < 0 || ih >= static_cast<ptrdiff_t>(IH)) {
                                continue;
                            }
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const ptrdiff_t iw = iw_base + static_cast<ptrdiff_t>(kw);
                                if (iw < 0 || iw >= static_cast<ptrdiff_t>(IW)) {
                                    continue;
                                }
                                const uint8_t* src_ptr =
                                    src_n + static_cast<size_t>(ih) * stride_h + static_cast<size_t>(iw) * stride_w;
                                const int8_t* wei_ptr =
                                    packed_wei_.data() + ((kh * KW + kw) * OC + oc) * IC;
                                if (use_dot_s8) {
                                    ker_block4_dot(reinterpret_cast<const int8_t*>(src_ptr),
                                                   wei_ptr,
                                                   dst_ptr,
                                                   IC,
                                                   IC,
                                                   wrote ? 1 : 0);
                                } else if (use_dot_u8) {
                                    ker_block4_udot(src_ptr, wei_ptr, dst_ptr, IC, IC, wrote ? 1 : 0);
                                } else if (src_signed) {
                                    ker_block4_s8(src_ptr, wei_ptr, dst_ptr, IC, IC, wrote ? 1 : 0);
                                } else {
                                    ker_block4_u8(src_ptr, wei_ptr, dst_ptr, IC, IC, wrote ? 1 : 0);
                                }
                                wrote = true;
                            }
                        }
                        if (!wrote) {
                            dst_ptr[0] = bias_ptr ? bias_ptr[oc] : 0;
                            dst_ptr[1] = bias_ptr ? bias_ptr[oc + 1] : 0;
                            dst_ptr[2] = bias_ptr ? bias_ptr[oc + 2] : 0;
                            dst_ptr[3] = bias_ptr ? bias_ptr[oc + 3] : 0;
                        } else {
                            if (use_dot_u8 && !wei_comp_brgemm_.empty()) {
                                dst_ptr[0] += wei_comp_brgemm_[oc];
                                dst_ptr[1] += wei_comp_brgemm_[oc + 1];
                                dst_ptr[2] += wei_comp_brgemm_[oc + 2];
                                dst_ptr[3] += wei_comp_brgemm_[oc + 3];
                            }
                            if (bias_ptr) {
                                dst_ptr[0] += bias_ptr[oc];
                                dst_ptr[1] += bias_ptr[oc + 1];
                                dst_ptr[2] += bias_ptr[oc + 2];
                                dst_ptr[3] += bias_ptr[oc + 3];
                            }
                        }
                    }
                    for (; oc < OC; oc++) {
                        int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                        bool wrote = false;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh);
                            if (ih < 0 || ih >= static_cast<ptrdiff_t>(IH)) {
                                continue;
                            }
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const ptrdiff_t iw = iw_base + static_cast<ptrdiff_t>(kw);
                                if (iw < 0 || iw >= static_cast<ptrdiff_t>(IW)) {
                                    continue;
                                }
                                const uint8_t* src_ptr =
                                    src_n + static_cast<size_t>(ih) * stride_h + static_cast<size_t>(iw) * stride_w;
                                const int8_t* wei_ptr =
                                    packed_wei_.data() + ((kh * KW + kw) * OC + oc) * IC;
                                ker(reinterpret_cast<const uint8_t*>(src_ptr), wei_ptr, dst_ptr, IC, wrote ? 1 : 0);
                                wrote = true;
                            }
                        }
                        if (!wrote) {
                            *dst_ptr = bias_ptr ? bias_ptr[oc] : 0;
                        } else if (bias_ptr) {
                            *dst_ptr += bias_ptr[oc];
                        }
                    }
                    ow++;
                }
        });
    }
}

void BrgemmInt8ConvExecutor::execute(const MemoryArgs& memory) {
    PlainTensor dst_out(memory.at(ARG_DST));
    const auto* fq = getFqPostOp(attrs_.postOps);
    const bool dst_i32 = dst_out.get_precision() == ov::element::i32;
    const auto bias_prec = (bias_prec_ == ov::element::dynamic && attrs_.withBias)
                               ? memory.at(ARG_BIAS)->getDescPtr()->getPrecision()
                               : bias_prec_;
    const bool bias_fp = any_of(bias_prec, ov::element::f32, ov::element::f16);
    const std::vector<float>* bias_fp_ptr = bias_fp ? &bias_f32_ : nullptr;
    PlainTensor* dst_calc = &dst_out;

    if (fq || !dst_i32) {
        const auto& dims = dst_out.shape();
        std::vector<size_t> strides(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            strides[i] = dst_out.stride(i);
        }
        tmp_dst_.resize(dims, static_cast<int32_t*>(nullptr), strides.data());
        dst_calc = &tmp_dst_;
    }

    if (!execute_brgemm_1x1(*dst_calc, memory)) {
        execute_impl(*dst_calc, memory);
    }

    if (dst_calc != &dst_out) {
        if (fq) {
            if (dst_out.get_precision() == ov::element::u8) {
                requantize_fq<uint8_t>(*dst_calc, dst_out, attrs_.dqScales, *fq, bias_fp_ptr);
            } else if (dst_out.get_precision() == ov::element::i8) {
                requantize_fq<int8_t>(*dst_calc, dst_out, attrs_.dqScales, *fq, bias_fp_ptr);
            }
        } else {
            if (dst_out.get_precision() == ov::element::u8) {
                requantize_simple<uint8_t>(*dst_calc, dst_out, attrs_.dqScales, bias_fp_ptr);
            } else if (dst_out.get_precision() == ov::element::i8) {
                requantize_simple<int8_t>(*dst_calc, dst_out, attrs_.dqScales, bias_fp_ptr);
            }
        }
    }
}

}  // namespace ov::intel_cpu::aarch64
