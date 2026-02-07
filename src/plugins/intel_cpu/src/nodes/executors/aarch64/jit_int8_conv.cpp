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

#if defined(__linux__)
#    include <sys/auxv.h>
#endif
#if defined(__linux__) && defined(__aarch64__)
#    include <asm/hwcap.h>
#endif

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

bool has_asimd_dotprod() {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}

size_t packed_block_stride(size_t K, size_t oc_block) {
    const size_t k_blocks = K / 16;
    const size_t k_tail = K % 16;
    return k_blocks * oc_block * 16 + k_tail * oc_block;
}

size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
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
      kernel_block4x4_dot_s8_(std::make_shared<jit_int8_brgemm_kernel_4x4_dot>()),
      kernel_block4x4_mmla_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_4x4_smmla_packed>()),
      kernel_block4x8_mmla_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_4x8_smmla_packed>()),
      kernel_block4x16_mmla_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_4x16_smmla_packed>()),
      kernel_block4x4_udot_(std::make_shared<jit_int8_brgemm_kernel_4x4_udot>()),
      kernel_block4x4_dot_packed_s8_(std::make_shared<jit_int8_brgemm_kernel_4x4_dot_packed>()),
      kernel_block4x4_mmla_packed_u8_(std::make_shared<jit_int8_brgemm_kernel_4x4_usmmla_packed>()),
      kernel_block4x8_mmla_packed_u8_(std::make_shared<jit_int8_brgemm_kernel_4x8_usmmla_packed>()),
      kernel_block4x16_mmla_packed_u8_(std::make_shared<jit_int8_brgemm_kernel_4x16_usmmla_packed>()),
      kernel_block4x4_udot_packed_(std::make_shared<jit_int8_brgemm_kernel_4x4_udot_packed>()),
      has_dotprod_(has_asimd_dotprod()),
      has_i8mm_(ov::intel_cpu::hasInt8MMSupport()) {
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
    kernel_block4x4_dot_s8_->create_ker();
    kernel_block4x4_mmla_packed_s8_->create_ker();
    kernel_block4x8_mmla_packed_s8_->create_ker();
    kernel_block4x16_mmla_packed_s8_->create_ker();
    kernel_block4x4_udot_->create_ker();
    kernel_block4x4_dot_packed_s8_->create_ker();
    kernel_block4x4_mmla_packed_u8_->create_ker();
    kernel_block4x8_mmla_packed_u8_->create_ker();
    kernel_block4x16_mmla_packed_u8_->create_ker();
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
            packed_wei_1x1_.assign(IC * OC, 0);
            packed_wei_1x1_col_.assign(OC * IC, 0);
            packed_wei_1x1_dot4_.clear();
            packed_wei_1x1_dot8_.clear();
            packed_wei_1x1_mmla4_.clear();
            packed_wei_1x1_mmla8_.clear();
            packed_wei_1x1_mmla16_.clear();
            packed_wei_1x1_dot4_stride_ = 0;
            packed_wei_1x1_dot8_stride_ = 0;
            packed_wei_1x1_mmla4_stride_ = 0;
            packed_wei_1x1_mmla8_stride_ = 0;
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
            if (has_dotprod_) {
                const size_t oc_blocks8 = OC / 8;
                const size_t oc_blocks4 = OC / 4;
                packed_wei_1x1_dot8_stride_ = packed_block_stride(IC, 8);
                packed_wei_1x1_dot4_stride_ = packed_block_stride(IC, 4);
                if (oc_blocks8 > 0) {
                    packed_wei_1x1_dot8_.assign(oc_blocks8 * packed_wei_1x1_dot8_stride_, 0);
                    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
                        const int8_t* src_block = packed_wei_1x1_col_.data() + ocb * 8 * IC;
                        int8_t* dst_block = packed_wei_1x1_dot8_.data() + ocb * packed_wei_1x1_dot8_stride_;
                        pack_dot_block(src_block, IC, 8, dst_block);
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
            if (has_i8mm_ && (IC % 8 == 0)) {
                const size_t ic_padded = round_up(IC, 8);
                const size_t oc_blocks16 = OC / 16;
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
    } else {
        packed_wei_1x1_.clear();
        packed_wei_1x1_col_.clear();
        packed_wei_1x1_dot4_.clear();
        packed_wei_1x1_dot8_.clear();
        packed_wei_1x1_mmla4_.clear();
        packed_wei_1x1_mmla8_.clear();
        packed_wei_1x1_mmla16_.clear();
        packed_wei_1x1_dot4_stride_ = 0;
        packed_wei_1x1_dot8_stride_ = 0;
        packed_wei_1x1_mmla4_stride_ = 0;
        packed_wei_1x1_mmla8_stride_ = 0;
        packed_wei_1x1_mmla16_stride_ = 0;
        wei_comp_1x1_.clear();
        packed_wei_1x1_src_ = nullptr;
        packed_wei_1x1_oc_ = 0;
        packed_wei_1x1_ic_ = 0;
    }

    if (KH != 1 || KW != 1) {
        const void* wei_ptr_base = wei.ptr<int8_t>(0, 0, 0, 0);
        if (packed_wei_brgemm_src_ != wei_ptr_base || packed_wei_brgemm_oc_ != OC || packed_wei_brgemm_ic_ != IC ||
            packed_wei_brgemm_kh_ != KH || packed_wei_brgemm_kw_ != KW) {
            packed_wei_brgemm_.assign(KH * KW * IC * OC, 0);
            packed_wei_brgemm_col_.assign(KH * KW * OC * IC, 0);
            packed_wei_brgemm_dot4_.clear();
            packed_wei_brgemm_dot8_.clear();
            packed_wei_brgemm_mmla4_.clear();
            packed_wei_brgemm_mmla8_.clear();
            packed_wei_brgemm_mmla16_.clear();
            packed_wei_brgemm_mmla4_fused_.clear();
            packed_wei_brgemm_mmla8_fused_.clear();
            packed_wei_brgemm_mmla16_fused_.clear();
            packed_wei_brgemm_dot4_stride_ = 0;
            packed_wei_brgemm_dot8_stride_ = 0;
            packed_wei_brgemm_mmla4_stride_ = 0;
            packed_wei_brgemm_mmla8_stride_ = 0;
            packed_wei_brgemm_mmla16_stride_ = 0;
            packed_wei_brgemm_mmla4_fused_stride_ = 0;
            packed_wei_brgemm_mmla8_fused_stride_ = 0;
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
            if (has_dotprod_) {
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
            if (has_i8mm_ && (IC % 8 == 0)) {
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
            if (has_i8mm_) {
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
        packed_wei_brgemm_mmla16_fused_.clear();
        packed_wei_brgemm_dot4_stride_ = 0;
        packed_wei_brgemm_dot8_stride_ = 0;
        packed_wei_brgemm_mmla4_stride_ = 0;
        packed_wei_brgemm_mmla8_stride_ = 0;
        packed_wei_brgemm_mmla16_stride_ = 0;
        packed_wei_brgemm_mmla4_fused_stride_ = 0;
        packed_wei_brgemm_mmla8_fused_stride_ = 0;
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

    const size_t N = src_dims[0];
    const size_t IC = src_dims[1];
    const size_t IH = src_dims[2];
    const size_t IW = src_dims[3];
    const size_t OH = dst_dims[2];
    const size_t OW = dst_dims[3];
    const size_t OC = dst_dims[1];

    const size_t stride_h = attrs_.stride[0];
    const size_t stride_w = attrs_.stride[1];
    const ptrdiff_t pad_t = attrs_.paddingL[0];
    const ptrdiff_t pad_l = attrs_.paddingL[1];
    const ptrdiff_t pad_r = attrs_.paddingR[1];

    const size_t lda = src.stride(3) * stride_w;
    const size_t ldc = dst.stride(3);
    if (lda < IC || ldc < OC) {
        return false;
    }

    const bool src_signed = src.get_precision() == ov::element::i8;
    auto& brgemm_kernel = src_signed ? brgemm_1x1_s8_ : brgemm_1x1_u8_;
    if (!brgemm_kernel || brgemm_1x1_oc_ != OC || brgemm_1x1_ic_ != IC || brgemm_1x1_lda_ != lda ||
        brgemm_1x1_ldc_ != ldc) {
        brgemm_kernel = std::make_shared<BrgemmInt8Kernel>(brgemm_1x1_m_blk_, OC, IC, lda, OC, ldc, src_signed);
        brgemm_1x1_oc_ = OC;
        brgemm_1x1_ic_ = IC;
        brgemm_1x1_lda_ = lda;
        brgemm_1x1_ldc_ = ldc;
    }
    const bool use_comp = !src_signed && brgemm_kernel && !brgemm_kernel->uses_brgemm() && !wei_comp_1x1_.empty();
    const int32_t* comp_ptr = use_comp ? wei_comp_1x1_.data() : nullptr;

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
    const auto ker_block4x4_dot = kernel_block4x4_dot_s8_->ker();
    const auto ker_block4x4_mmla_packed_s8 = kernel_block4x4_mmla_packed_s8_->ker();
    const auto ker_block4x8_mmla_packed_s8 = kernel_block4x8_mmla_packed_s8_->ker();
    const auto ker_block4x16_mmla_packed_s8 = kernel_block4x16_mmla_packed_s8_->ker();
    const auto ker_block4x4_udot = kernel_block4x4_udot_->ker();
    const auto ker_block4x4_dot_packed = kernel_block4x4_dot_packed_s8_->ker();
    const auto ker_block4x4_mmla_packed_u8 = kernel_block4x4_mmla_packed_u8_->ker();
    const auto ker_block4x8_mmla_packed_u8 = kernel_block4x8_mmla_packed_u8_->ker();
    const auto ker_block4x16_mmla_packed_u8 = kernel_block4x16_mmla_packed_u8_->ker();
    const auto ker_block4x4_udot_packed = kernel_block4x4_udot_packed_->ker();
    const bool use_dot_s8 = src_signed && has_dotprod_;
    const bool use_dot_u8 = !src_signed && has_dotprod_;
    const bool use_mmla_s8 = src_signed && has_i8mm_;
    const bool use_mmla_u8 = !src_signed && has_i8mm_;

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
    const bool use_block4x4_dot = use_dot_s8 && (KH == 1) && (KW == 1);
    const bool use_block4x4_udot = use_dot_u8 && (KH == 1) && (KW == 1);
    const bool use_block4x16_mmla =
        (use_mmla_s8 || use_mmla_u8) && (KH == 1) && (KW == 1) && (IC % 8 == 0) &&
        !packed_wei_1x1_mmla16_.empty();
    const bool use_block4x8_mmla =
        (use_mmla_s8 || use_mmla_u8) && (KH == 1) && (KW == 1) && (IC % 8 == 0) &&
        !packed_wei_1x1_mmla8_.empty();
    const bool use_block4x4_mmla =
        (use_mmla_s8 || use_mmla_u8) && (KH == 1) && (KW == 1) && (IC % 8 == 0) &&
        !packed_wei_1x1_mmla4_.empty();
    const bool use_packed4_1x1 = has_dotprod_ && !packed_wei_1x1_dot4_.empty();
    const bool use_packed8_1x1 = has_dotprod_ && !packed_wei_1x1_dot8_.empty();
    const size_t conv_stride_h = attrs_.stride[0];
    const size_t conv_stride_w = attrs_.stride[1];
    const ptrdiff_t pad_t = attrs_.paddingL[0];
    const ptrdiff_t pad_l = attrs_.paddingL[1];

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

    if (KH == 1 && KW == 1) {
        for (size_t n = 0; n < N; n++) {
            for (size_t oh = 0; oh < OH; oh++) {
                const ptrdiff_t ih = static_cast<ptrdiff_t>(oh * conv_stride_h) - pad_t;
                size_t ow = 0;
                if ((use_block4x4_dot || use_block4x4_udot || use_block4x4_mmla || use_block4x8_mmla ||
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
                                if (use_mmla_u8 && !wei_comp_1x1_.empty()) {
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
                                    if (use_mmla_u8 && !wei_comp_1x1_.empty()) {
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
                                    if (use_mmla_u8 && !wei_comp_1x1_.empty()) {
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
                                    if (!wei_comp_1x1_.empty()) {
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
                                            if (!wei_comp_1x1_.empty()) {
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
                                if (!wei_comp_1x1_.empty()) {
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
                                if (!wei_comp_1x1_.empty()) {
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
        const bool use_kxk_mmla_fused = use_packed16_kxk_mmla_fused || use_packed8_kxk_mmla_fused ||
                                        use_packed4_kxk_mmla_fused;
        const size_t lda_brgemm = stride_w * conv_stride_w;
        const size_t ldc_brgemm = dst.stride(3);
        const bool contiguous_w = (stride_w == IC) && (conv_stride_w == 1);
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

        const size_t ow_step = brgemm_1x1_m_blk_;
        const size_t ow_blocks = (OW + ow_step - 1) / ow_step;
        parallel_for3d(N, OH, ow_blocks, [&](size_t n, size_t oh, size_t owb) {
            static thread_local std::vector<uint8_t> packed_src_u8_tls;
            static thread_local std::vector<int8_t> packed_src_s8_tls;
            if (use_kxk_mmla_fused) {
                const size_t pack_elems = brgemm_1x1_m_blk_ * kxk_kp;
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
                    const bool full_w =
                        (use_dot_s8 || use_dot_u8 || use_packed4_kxk_mmla || use_packed8_kxk_mmla ||
                         use_packed16_kxk_mmla) &&
                        full_h && (ow + 4 <= ow_end) && iw_base >= 0 &&
                        (iw_base + static_cast<ptrdiff_t>(KW - 1 + 3 * conv_stride_w)) <
                            static_cast<ptrdiff_t>(IW);
                    const bool full_w_fused =
                        use_kxk_mmla_fused && full_h && (ow + 4 <= ow_end) && iw_base >= 0 &&
                        (iw_base + static_cast<ptrdiff_t>(KW - 1 + 3 * conv_stride_w)) <
                            static_cast<ptrdiff_t>(IW);
                    if (full_w_fused) {
                        uint8_t* packed_src_u8 = packed_src_u8_tls.empty() ? nullptr : packed_src_u8_tls.data();
                        int8_t* packed_src_s8 = packed_src_s8_tls.empty() ? nullptr : packed_src_s8_tls.data();
                        if (src_signed) {
                            int8_t* pack_ptr = packed_src_s8;
                            for (size_t m = 0; m < 4; ++m) {
                                const size_t base = m * kxk_kp;
                                if (kxk_kp != kxk_k) {
                                    std::memset(pack_ptr + base, 0, kxk_kp);
                                }
                                for (size_t kh = 0; kh < KH; ++kh) {
                                    const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh));
                                    const size_t src_h_off = ih * stride_h;
                                    if (contiguous_w) {
                                        const size_t iw =
                                            static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(m * conv_stride_w));
                                        const uint8_t* src_ptr_u8 = src_n + src_h_off + iw * stride_w;
                                        const int8_t* src_ptr = reinterpret_cast<const int8_t*>(src_ptr_u8);
                                        const size_t dst_off = base + kh * KW * IC;
                                        std::memcpy(pack_ptr + dst_off, src_ptr, KW * IC);
                                    } else {
                                        for (size_t kw = 0; kw < KW; ++kw) {
                                            const size_t iw =
                                                static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(kw) +
                                                                    static_cast<ptrdiff_t>(m * conv_stride_w));
                                            const uint8_t* src_ptr_u8 = src_n + src_h_off + iw * stride_w;
                                            const int8_t* src_ptr = reinterpret_cast<const int8_t*>(src_ptr_u8);
                                            const size_t dst_off = base + (kh * KW + kw) * IC;
                                            std::copy_n(src_ptr, IC, pack_ptr + dst_off);
                                        }
                                    }
                                }
                            }
                        } else {
                            uint8_t* pack_ptr = packed_src_u8;
                            for (size_t m = 0; m < 4; ++m) {
                                const size_t base = m * kxk_kp;
                                if (kxk_kp != kxk_k) {
                                    std::memset(pack_ptr + base, 0, kxk_kp);
                                }
                                for (size_t kh = 0; kh < KH; ++kh) {
                                    const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh));
                                    const size_t src_h_off = ih * stride_h;
                                    if (contiguous_w) {
                                        const size_t iw =
                                            static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(m * conv_stride_w));
                                        const uint8_t* src_ptr = src_n + src_h_off + iw * stride_w;
                                        const size_t dst_off = base + kh * KW * IC;
                                        std::memcpy(pack_ptr + dst_off, src_ptr, KW * IC);
                                    } else {
                                        for (size_t kw = 0; kw < KW; ++kw) {
                                            const size_t iw =
                                                static_cast<size_t>(iw_base + static_cast<ptrdiff_t>(kw) +
                                                                    static_cast<ptrdiff_t>(m * conv_stride_w));
                                            const uint8_t* src_ptr = src_n + src_h_off + iw * stride_w;
                                            const size_t dst_off = base + (kh * KW + kw) * IC;
                                            std::copy_n(src_ptr, IC, pack_ptr + dst_off);
                                        }
                                    }
                                }
                            }
                        }

                        size_t oc = 0;
                        const int8_t* src_ptrs_s8[4] = {
                            packed_src_s8,
                            packed_src_s8 ? packed_src_s8 + 1 * kxk_kp : nullptr,
                            packed_src_s8 ? packed_src_s8 + 2 * kxk_kp : nullptr,
                            packed_src_s8 ? packed_src_s8 + 3 * kxk_kp : nullptr,
                        };
                        const uint8_t* src_ptrs_u8[4] = {
                            packed_src_u8,
                            packed_src_u8 ? packed_src_u8 + 1 * kxk_kp : nullptr,
                            packed_src_u8 ? packed_src_u8 + 2 * kxk_kp : nullptr,
                            packed_src_u8 ? packed_src_u8 + 3 * kxk_kp : nullptr,
                        };
                        for (; use_packed16_kxk_mmla_fused && oc + 16 <= OC; oc += 16) {
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            const int8_t* wei_ptr =
                                packed_wei_brgemm_mmla16_fused_.data() +
                                (oc / 16) * packed_wei_brgemm_mmla16_fused_stride_;
                            if (use_mmla_s8) {
                                ker_block4x16_mmla_packed_s8(src_ptrs_s8,
                                                             wei_ptr,
                                                             dst_ptr,
                                                             kxk_kp,
                                                             0,
                                                             OC * sizeof(int32_t),
                                                             0);
                            } else {
                                ker_block4x16_mmla_packed_u8(src_ptrs_u8,
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
                        for (; use_packed8_kxk_mmla_fused && oc + 8 <= OC; oc += 8) {
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            const int8_t* wei_ptr =
                                packed_wei_brgemm_mmla8_fused_.data() +
                                (oc / 8) * packed_wei_brgemm_mmla8_fused_stride_;
                            if (use_mmla_s8) {
                                ker_block4x8_mmla_packed_s8(src_ptrs_s8,
                                                            wei_ptr,
                                                            dst_ptr,
                                                            kxk_kp,
                                                            0,
                                                            OC * sizeof(int32_t),
                                                            0);
                            } else {
                                ker_block4x8_mmla_packed_u8(src_ptrs_u8,
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
                        for (; use_packed4_kxk_mmla_fused && oc + 4 <= OC; oc += 4) {
                            int32_t* dst_ptr = dst.ptr<int32_t>(n, oc, oh, ow);
                            const int8_t* wei_ptr =
                                packed_wei_brgemm_mmla4_fused_.data() +
                                (oc / 4) * packed_wei_brgemm_mmla4_fused_stride_;
                            if (use_mmla_s8) {
                                ker_block4x4_mmla_packed_s8(src_ptrs_s8,
                                                            wei_ptr,
                                                            dst_ptr,
                                                            kxk_kp,
                                                            0,
                                                            OC * sizeof(int32_t),
                                                            0);
                            } else {
                                ker_block4x4_mmla_packed_u8(src_ptrs_u8,
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
