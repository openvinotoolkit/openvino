// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_ARM64)

#include "nodes/kernels/aarch64/jit_uni_topk_kernel.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if !defined(HAVE_SVE) && defined(__ARM_FEATURE_SVE)
#    define HAVE_SVE 1
#endif
#if !defined(HAVE_SVE2) && defined(__ARM_FEATURE_SVE2)
#    define HAVE_SVE2 1
#endif
#include <arm_neon.h>
#if defined(HAVE_SVE)
#    include <arm_sve.h>
#endif

#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::node {
namespace {
template <typename T>
struct TopkCompare {
    static inline bool lt(const T& a, const T& b) {
        return a < b;
    }
    static inline bool gt(const T& a, const T& b) {
        return a > b;
    }
};

template <>
struct TopkCompare<ov::float16> {
    static inline bool lt(const ov::float16& a, const ov::float16& b) {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    static inline bool gt(const ov::float16& a, const ov::float16& b) {
        return static_cast<float>(a) > static_cast<float>(b);
    }
};

inline uint32x4_t expand_mask_u16(uint16x8_t mask16, bool high) {
    uint32x4_t m32 = vmovl_u16(high ? vget_high_u16(mask16) : vget_low_u16(mask16));
    return vorrq_u32(m32, vshlq_n_u32(m32, 16));
}

inline void expand_mask_u8(uint8x16_t mask8,
                           uint32x4_t& m0,
                           uint32x4_t& m1,
                           uint32x4_t& m2,
                           uint32x4_t& m3) {
    const uint16x8_t m16_low = vmovl_u8(vget_low_u8(mask8));
    const uint16x8_t m16_high = vmovl_u8(vget_high_u8(mask8));

    m0 = vmovl_u16(vget_low_u16(m16_low));
    m1 = vmovl_u16(vget_high_u16(m16_low));
    m2 = vmovl_u16(vget_low_u16(m16_high));
    m3 = vmovl_u16(vget_high_u16(m16_high));

    auto expand = [](uint32x4_t m) {
        m = vorrq_u32(m, vshlq_n_u32(m, 8));
        m = vorrq_u32(m, vshlq_n_u32(m, 16));
        return m;
    };

    m0 = expand(m0);
    m1 = expand(m1);
    m2 = expand(m2);
    m3 = expand(m3);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline float16x8_t load_f16x8(const ov::float16* src) {
    float16_t tmp[8];
    std::memcpy(tmp, src, sizeof(tmp));
    return vld1q_f16(tmp);
}

inline void store_f16x8(ov::float16* dst, float16x8_t v) {
    float16_t tmp[8];
    vst1q_f16(tmp, v);
    std::memcpy(dst, tmp, sizeof(tmp));
}
#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(HAVE_SVE)
inline svfloat32_t sve_cvt_f32_f16_low(svbool_t pg_f32, svfloat16_t v) {
    return svcvt_f32_f16_x(pg_f32, svzip1_f16(v, v));
}

inline svfloat32_t sve_cvt_f32_f16_high(svbool_t pg_f32, svfloat16_t v) {
    return svcvt_f32_f16_x(pg_f32, svzip2_f16(v, v));
}

inline svfloat16_t sve_pack_f16_from_f32(svbool_t pg_f32, svfloat32_t low, svfloat32_t high) {
    const svfloat16_t lo = svcvt_f16_f32_x(pg_f32, low);
    const svfloat16_t hi = svcvt_f16_f32_x(pg_f32, high);
    return svuzp1_f16(lo, hi);
}
#endif  // HAVE_SVE

template <typename T, size_t StackMax>
struct TopkBuffers {
    T* vals = nullptr;
    int32_t* idx = nullptr;
    std::array<T, StackMax + 1> vals_stack{};
    std::array<int32_t, StackMax + 1> idx_stack{};
    std::vector<T> vals_dyn;
    std::vector<int32_t> idx_dyn;

    explicit TopkBuffers(size_t top_k) {
        if (top_k <= StackMax) {
            vals = vals_stack.data();
            idx = idx_stack.data();
        } else {
            vals_dyn.resize(top_k + 1);
            idx_dyn.resize(top_k + 1);
            vals = vals_dyn.data();
            idx = idx_dyn.data();
        }
    }
};

template <typename T>
inline T load_value(const T* base,
                    size_t axis_idx,
                    size_t lane,
                    const jit_topk_config_params* jcp,
                    const jit_topk_call_args* args) {
    if (jcp->layout == TopKLayoutType::topk_blocked && jcp->topk_innermost) {
        const size_t blk_stride = args->sort_stride * jcp->blk_size;
        const size_t block = axis_idx / jcp->blk_size;
        const size_t offset = axis_idx % jcp->blk_size;
        const size_t idx = block * blk_stride + offset + lane * jcp->blk_size;
        return base[idx];
    } else {
        const size_t idx = axis_idx * args->sort_stride + lane;
        return base[idx];
    }
}

template <typename T>
inline void store_value(T* base,
                        size_t axis_idx,
                        size_t lane,
                        const jit_topk_config_params* jcp,
                        const jit_topk_call_args* args,
                        const T& value) {
    if (jcp->layout == TopKLayoutType::topk_blocked && jcp->topk_innermost) {
        const size_t blk_stride = args->sort_stride * jcp->blk_size;
        const size_t block = axis_idx / jcp->blk_size;
        const size_t offset = axis_idx % jcp->blk_size;
        const size_t idx = block * blk_stride + offset + lane * jcp->blk_size;
        base[idx] = value;
    } else {
        const size_t idx = axis_idx * args->sort_stride + lane;
        base[idx] = value;
    }
}

inline void store_index(int32_t* base,
                        size_t axis_idx,
                        size_t lane,
                        const jit_topk_config_params* jcp,
                        const jit_topk_call_args* args,
                        int32_t value) {
    if (jcp->layout == TopKLayoutType::topk_blocked && jcp->topk_innermost) {
        const size_t blk_stride = args->sort_stride * jcp->blk_size;
        const size_t block = axis_idx / jcp->blk_size;
        const size_t offset = axis_idx % jcp->blk_size;
        const size_t idx = block * blk_stride + offset + lane * jcp->blk_size;
        base[idx] = value;
    } else {
        const size_t idx = axis_idx * args->sort_stride + lane;
        base[idx] = value;
    }
}

template <typename T>
void topk_kernel_impl(const jit_topk_call_args* args) {
    const auto* jcp = args->config;
    const auto* src = static_cast<const T*>(args->src);
    auto* dst = static_cast<T*>(args->dst);
    auto* dst_idx = static_cast<int32_t*>(args->index);

    const size_t axis_dim = args->axis_dim;
    const size_t top_k = args->top_k;
    const size_t work_amount = args->work_amount;

    if (axis_dim == 0 || top_k == 0 || work_amount == 0) {
        return;
    }

    if (top_k == 1) {
        const bool mode_max = jcp->mode_max;
        if (!jcp->topk_innermost) {
            size_t lane = 0;
#if defined(HAVE_SVE)
            if constexpr (std::is_same_v<T, float>) {
                if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128)) {
                    const size_t vlen = svcntw();
                    auto process_vec = [&](svbool_t pg, size_t lane_base) {
                        const auto* base = reinterpret_cast<const float*>(src + lane_base);
                        svfloat32_t best = svld1(pg, base);
                        svint32_t best_idx = svdup_s32(0);
                        const size_t stride = args->sort_stride;
                        const T* ptr = src + stride + lane_base;
                        size_t i = 1;
                        for (; i + 3 < axis_dim; i += 4) {
                            const auto* ptr0 = reinterpret_cast<const float*>(ptr);
                            const auto* ptr1 = reinterpret_cast<const float*>(ptr + stride);
                            const auto* ptr2 = reinterpret_cast<const float*>(ptr + 2 * stride);
                            const auto* ptr3 = reinterpret_cast<const float*>(ptr + 3 * stride);
                            const svfloat32_t v0 = svld1(pg, ptr0);
                            const svfloat32_t v1 = svld1(pg, ptr1);
                            const svfloat32_t v2 = svld1(pg, ptr2);
                            const svfloat32_t v3 = svld1(pg, ptr3);

                            svbool_t m0 = mode_max ? svcmpgt_f32(pg, v0, best) : svcmplt_f32(pg, v0, best);
                            best = svsel(m0, v0, best);
                            best_idx = svsel(m0, svdup_s32(static_cast<int32_t>(i + 0)), best_idx);

                            svbool_t m1 = mode_max ? svcmpgt_f32(pg, v1, best) : svcmplt_f32(pg, v1, best);
                            best = svsel(m1, v1, best);
                            best_idx = svsel(m1, svdup_s32(static_cast<int32_t>(i + 1)), best_idx);

                            svbool_t m2 = mode_max ? svcmpgt_f32(pg, v2, best) : svcmplt_f32(pg, v2, best);
                            best = svsel(m2, v2, best);
                            best_idx = svsel(m2, svdup_s32(static_cast<int32_t>(i + 2)), best_idx);

                            svbool_t m3 = mode_max ? svcmpgt_f32(pg, v3, best) : svcmplt_f32(pg, v3, best);
                            best = svsel(m3, v3, best);
                            best_idx = svsel(m3, svdup_s32(static_cast<int32_t>(i + 3)), best_idx);
                            ptr += 4 * stride;
                        }
                        for (; i < axis_dim; ++i) {
                            const auto* ptr_f = reinterpret_cast<const float*>(ptr);
                            const svfloat32_t v = svld1(pg, ptr_f);
                            svbool_t mask = mode_max ? svcmpgt_f32(pg, v, best) : svcmplt_f32(pg, v, best);
                            best = svsel(mask, v, best);
                            svint32_t idx_vec = svdup_s32(static_cast<int32_t>(i));
                            best_idx = svsel(mask, idx_vec, best_idx);
                            ptr += stride;
                        }
                        auto* out = reinterpret_cast<float*>(dst + lane_base);
                        auto* out_idx = dst_idx + lane_base;
                        svst1(pg, out, best);
                        svst1(pg, out_idx, best_idx);
                    };

                    for (; lane + vlen <= work_amount; lane += vlen) {
                        process_vec(svptrue_b32(), lane);
                    }
                    if (lane < work_amount) {
                        process_vec(svwhilelt_b32(lane, work_amount), lane);
                    }
                    return;
                }
            }
            if constexpr (std::is_same_v<T, int32_t>) {
                if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128)) {
                    const size_t vlen = svcntw();
                    auto process_vec = [&](svbool_t pg, size_t lane_base) {
                        const auto* base = reinterpret_cast<const int32_t*>(src + lane_base);
                        svint32_t best = svld1(pg, base);
                        svint32_t best_idx = svdup_s32(0);
                        const size_t stride = args->sort_stride;
                        const T* ptr = src + stride + lane_base;
                        size_t i = 1;
                        for (; i + 3 < axis_dim; i += 4) {
                            const auto* ptr0 = reinterpret_cast<const int32_t*>(ptr);
                            const auto* ptr1 = reinterpret_cast<const int32_t*>(ptr + stride);
                            const auto* ptr2 = reinterpret_cast<const int32_t*>(ptr + 2 * stride);
                            const auto* ptr3 = reinterpret_cast<const int32_t*>(ptr + 3 * stride);
                            const svint32_t v0 = svld1(pg, ptr0);
                            const svint32_t v1 = svld1(pg, ptr1);
                            const svint32_t v2 = svld1(pg, ptr2);
                            const svint32_t v3 = svld1(pg, ptr3);

                            svbool_t m0 = mode_max ? svcmpgt_s32(pg, v0, best) : svcmplt_s32(pg, v0, best);
                            best = svsel(m0, v0, best);
                            best_idx = svsel(m0, svdup_s32(static_cast<int32_t>(i + 0)), best_idx);

                            svbool_t m1 = mode_max ? svcmpgt_s32(pg, v1, best) : svcmplt_s32(pg, v1, best);
                            best = svsel(m1, v1, best);
                            best_idx = svsel(m1, svdup_s32(static_cast<int32_t>(i + 1)), best_idx);

                            svbool_t m2 = mode_max ? svcmpgt_s32(pg, v2, best) : svcmplt_s32(pg, v2, best);
                            best = svsel(m2, v2, best);
                            best_idx = svsel(m2, svdup_s32(static_cast<int32_t>(i + 2)), best_idx);

                            svbool_t m3 = mode_max ? svcmpgt_s32(pg, v3, best) : svcmplt_s32(pg, v3, best);
                            best = svsel(m3, v3, best);
                            best_idx = svsel(m3, svdup_s32(static_cast<int32_t>(i + 3)), best_idx);
                            ptr += 4 * stride;
                        }
                        for (; i < axis_dim; ++i) {
                            const auto* ptr_i = reinterpret_cast<const int32_t*>(ptr);
                            const svint32_t v = svld1(pg, ptr_i);
                            const svbool_t mask = mode_max ? svcmpgt_s32(pg, v, best) : svcmplt_s32(pg, v, best);
                            best = svsel(mask, v, best);
                            svint32_t idx_vec = svdup_s32(static_cast<int32_t>(i));
                            best_idx = svsel(mask, idx_vec, best_idx);
                            ptr += stride;
                        }
                        auto* out = reinterpret_cast<int32_t*>(dst + lane_base);
                        auto* out_idx = dst_idx + lane_base;
                        svst1(pg, out, best);
                        svst1(pg, out_idx, best_idx);
                    };

                    for (; lane + vlen <= work_amount; lane += vlen) {
                        process_vec(svptrue_b32(), lane);
                    }
                    if (lane < work_amount) {
                        process_vec(svwhilelt_b32(lane, work_amount), lane);
                    }
                    return;
                }
            }
#endif
            if constexpr (std::is_same_v<T, float>) {
                constexpr size_t V = 4;
                for (; lane + V <= work_amount; lane += V) {
                    const auto* base = src + lane;
                    float32x4_t best = vld1q_f32(reinterpret_cast<const float*>(base));
                    int32x4_t best_idx = vdupq_n_s32(0);
                    const size_t stride = args->sort_stride;
                    const T* ptr = src + stride + lane;
                    size_t i = 1;
                    for (; i + 3 < axis_dim; i += 4) {
                        const auto* ptr0 = ptr;
                        const auto* ptr1 = ptr + stride;
                        const auto* ptr2 = ptr + 2 * stride;
                        const auto* ptr3 = ptr + 3 * stride;
                        float32x4_t v0 = vld1q_f32(reinterpret_cast<const float*>(ptr0));
                        float32x4_t v1 = vld1q_f32(reinterpret_cast<const float*>(ptr1));
                        float32x4_t v2 = vld1q_f32(reinterpret_cast<const float*>(ptr2));
                        float32x4_t v3 = vld1q_f32(reinterpret_cast<const float*>(ptr3));

                        const uint32x4_t m0 = mode_max ? vcgtq_f32(v0, best) : vcltq_f32(v0, best);
                        best = vbslq_f32(m0, v0, best);
                        best_idx = vbslq_s32(m0, vdupq_n_s32(static_cast<int32_t>(i + 0)), best_idx);

                        const uint32x4_t m1 = mode_max ? vcgtq_f32(v1, best) : vcltq_f32(v1, best);
                        best = vbslq_f32(m1, v1, best);
                        best_idx = vbslq_s32(m1, vdupq_n_s32(static_cast<int32_t>(i + 1)), best_idx);

                        const uint32x4_t m2 = mode_max ? vcgtq_f32(v2, best) : vcltq_f32(v2, best);
                        best = vbslq_f32(m2, v2, best);
                        best_idx = vbslq_s32(m2, vdupq_n_s32(static_cast<int32_t>(i + 2)), best_idx);

                        const uint32x4_t m3 = mode_max ? vcgtq_f32(v3, best) : vcltq_f32(v3, best);
                        best = vbslq_f32(m3, v3, best);
                        best_idx = vbslq_s32(m3, vdupq_n_s32(static_cast<int32_t>(i + 3)), best_idx);
                        ptr += 4 * stride;
                    }
                    for (; i < axis_dim; ++i) {
                        float32x4_t v = vld1q_f32(reinterpret_cast<const float*>(ptr));
                        const uint32x4_t mask =
                            mode_max ? vcgtq_f32(v, best) : vcltq_f32(v, best);
                        best = vbslq_f32(mask, v, best);
                        const int32x4_t idx_vec = vdupq_n_s32(static_cast<int32_t>(i));
                        best_idx = vbslq_s32(mask, idx_vec, best_idx);
                        ptr += stride;
                    }
                    auto* out = dst + lane;
                    auto* out_idx = dst_idx + lane;
                    vst1q_f32(reinterpret_cast<float*>(out), best);
                    vst1q_s32(out_idx, best_idx);
                }
            } else if constexpr (std::is_same_v<T, ov::float16>) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#    if defined(HAVE_SVE)
                if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128)) {
                    const size_t vlen_h = svcnth();
                    const size_t vlen_f32 = svcntw();
                    const svbool_t pg_f16 = svptrue_b16();
                    const svbool_t pg_f32 = svptrue_b32();
                    for (; lane + vlen_h <= work_amount; lane += vlen_h) {
                        const auto* base = reinterpret_cast<const float16_t*>(src + lane);
                        svfloat16_t best_f16 = svld1(pg_f16, base);
                        svfloat32_t best_low = sve_cvt_f32_f16_low(pg_f32, best_f16);
                        svfloat32_t best_high = sve_cvt_f32_f16_high(pg_f32, best_f16, vlen_f32);
                        svint32_t best_idx_low = svdup_s32(0);
                        svint32_t best_idx_high = svdup_s32(0);

                        auto update = [&](svfloat16_t v_f16, int32_t idx) {
                            svfloat32_t v_low = sve_cvt_f32_f16_low(pg_f32, v_f16);
                            svfloat32_t v_high = sve_cvt_f32_f16_high(pg_f32, v_f16, vlen_f32);
                            const svbool_t mask_low =
                                mode_max ? svcmpgt_f32(pg_f32, v_low, best_low)
                                         : svcmplt_f32(pg_f32, v_low, best_low);
                            const svbool_t mask_high =
                                mode_max ? svcmpgt_f32(pg_f32, v_high, best_high)
                                         : svcmplt_f32(pg_f32, v_high, best_high);
                            best_low = svsel(mask_low, v_low, best_low);
                            best_high = svsel(mask_high, v_high, best_high);
                            const svint32_t idx_vec = svdup_s32(idx);
                            best_idx_low = svsel(mask_low, idx_vec, best_idx_low);
                            best_idx_high = svsel(mask_high, idx_vec, best_idx_high);
                        };

                        const size_t stride = args->sort_stride;
                        const T* ptr = src + stride + lane;
                        size_t i = 1;
                        for (; i + 3 < axis_dim; i += 4) {
                            const auto* ptr0 = reinterpret_cast<const float16_t*>(ptr);
                            const auto* ptr1 = reinterpret_cast<const float16_t*>(ptr + stride);
                            const auto* ptr2 = reinterpret_cast<const float16_t*>(ptr + 2 * stride);
                            const auto* ptr3 = reinterpret_cast<const float16_t*>(ptr + 3 * stride);
                            update(svld1(pg_f16, ptr0), static_cast<int32_t>(i + 0));
                            update(svld1(pg_f16, ptr1), static_cast<int32_t>(i + 1));
                            update(svld1(pg_f16, ptr2), static_cast<int32_t>(i + 2));
                            update(svld1(pg_f16, ptr3), static_cast<int32_t>(i + 3));
                            ptr += 4 * stride;
                        }
                        for (; i < axis_dim; ++i) {
                            const auto* ptr_f16 = reinterpret_cast<const float16_t*>(ptr);
                            update(svld1(pg_f16, ptr_f16), static_cast<int32_t>(i));
                            ptr += stride;
                        }

                        auto* out = reinterpret_cast<float16_t*>(dst + lane);
                        auto* out_idx = dst_idx + lane;
                        svfloat16_t out_f16 = sve_pack_f16_from_f32(pg_f32, best_low, best_high, vlen_f32);
                        svst1(pg_f16, out, out_f16);
                        svst1(pg_f32, out_idx, best_idx_low);
                        svst1(pg_f32, out_idx + vlen_f32, best_idx_high);
                    }
                }
#    endif  // HAVE_SVE
                constexpr size_t V = 8;
                for (; lane + V <= work_amount; lane += V) {
                    const auto* base = src + lane;
                    float16x8_t best = load_f16x8(base);
                    int32x4_t best_idx0 = vdupq_n_s32(0);
                    int32x4_t best_idx1 = vdupq_n_s32(0);
                    const size_t stride = args->sort_stride;
                    const T* ptr = src + stride + lane;
                    auto update = [&](float16x8_t v, int32_t idx) {
                        const uint16x8_t mask = mode_max ? vcgtq_f16(v, best) : vcltq_f16(v, best);
                        best = vbslq_f16(mask, v, best);
                        const int32x4_t idx_vec = vdupq_n_s32(idx);
                        const uint32x4_t m0 = expand_mask_u16(mask, false);
                        const uint32x4_t m1 = expand_mask_u16(mask, true);
                        best_idx0 = vbslq_s32(m0, idx_vec, best_idx0);
                        best_idx1 = vbslq_s32(m1, idx_vec, best_idx1);
                    };

                    size_t i = 1;
                    for (; i + 3 < axis_dim; i += 4) {
                        const auto* ptr0 = ptr;
                        const auto* ptr1 = ptr + stride;
                        const auto* ptr2 = ptr + 2 * stride;
                        const auto* ptr3 = ptr + 3 * stride;
                        update(load_f16x8(ptr0), static_cast<int32_t>(i + 0));
                        update(load_f16x8(ptr1), static_cast<int32_t>(i + 1));
                        update(load_f16x8(ptr2), static_cast<int32_t>(i + 2));
                        update(load_f16x8(ptr3), static_cast<int32_t>(i + 3));
                        ptr += 4 * stride;
                    }
                    for (; i < axis_dim; ++i) {
                        update(load_f16x8(ptr), static_cast<int32_t>(i));
                        ptr += stride;
                    }
                    auto* out = dst + lane;
                    auto* out_idx = dst_idx + lane;
                    store_f16x8(out, best);
                    vst1q_s32(out_idx, best_idx0);
                    vst1q_s32(out_idx + 4, best_idx1);
                }
#endif
            } else if constexpr (std::is_same_v<T, int32_t>) {
                constexpr size_t V = 4;
                for (; lane + V <= work_amount; lane += V) {
                    const auto* base = src + lane;
                    int32x4_t best = vld1q_s32(reinterpret_cast<const int32_t*>(base));
                    int32x4_t best_idx = vdupq_n_s32(0);
                    const size_t stride = args->sort_stride;
                    const T* ptr = src + stride + lane;
                    size_t i = 1;
                    for (; i + 3 < axis_dim; i += 4) {
                        const auto* ptr0 = ptr;
                        const auto* ptr1 = ptr + stride;
                        const auto* ptr2 = ptr + 2 * stride;
                        const auto* ptr3 = ptr + 3 * stride;
                        int32x4_t v0 = vld1q_s32(reinterpret_cast<const int32_t*>(ptr0));
                        int32x4_t v1 = vld1q_s32(reinterpret_cast<const int32_t*>(ptr1));
                        int32x4_t v2 = vld1q_s32(reinterpret_cast<const int32_t*>(ptr2));
                        int32x4_t v3 = vld1q_s32(reinterpret_cast<const int32_t*>(ptr3));

                        const uint32x4_t m0 = mode_max ? vcgtq_s32(v0, best) : vcltq_s32(v0, best);
                        best = vbslq_s32(m0, v0, best);
                        best_idx = vbslq_s32(m0, vdupq_n_s32(static_cast<int32_t>(i + 0)), best_idx);

                        const uint32x4_t m1 = mode_max ? vcgtq_s32(v1, best) : vcltq_s32(v1, best);
                        best = vbslq_s32(m1, v1, best);
                        best_idx = vbslq_s32(m1, vdupq_n_s32(static_cast<int32_t>(i + 1)), best_idx);

                        const uint32x4_t m2 = mode_max ? vcgtq_s32(v2, best) : vcltq_s32(v2, best);
                        best = vbslq_s32(m2, v2, best);
                        best_idx = vbslq_s32(m2, vdupq_n_s32(static_cast<int32_t>(i + 2)), best_idx);

                        const uint32x4_t m3 = mode_max ? vcgtq_s32(v3, best) : vcltq_s32(v3, best);
                        best = vbslq_s32(m3, v3, best);
                        best_idx = vbslq_s32(m3, vdupq_n_s32(static_cast<int32_t>(i + 3)), best_idx);
                        ptr += 4 * stride;
                    }
                    for (; i < axis_dim; ++i) {
                        int32x4_t v = vld1q_s32(reinterpret_cast<const int32_t*>(ptr));
                        const uint32x4_t mask =
                            mode_max ? vcgtq_s32(v, best) : vcltq_s32(v, best);
                        best = vbslq_s32(mask, v, best);
                        const int32x4_t idx_vec = vdupq_n_s32(static_cast<int32_t>(i));
                        best_idx = vbslq_s32(mask, idx_vec, best_idx);
                        ptr += stride;
                    }
                    auto* out = dst + lane;
                    auto* out_idx = dst_idx + lane;
                    vst1q_s32(reinterpret_cast<int32_t*>(out), best);
                    vst1q_s32(out_idx, best_idx);
                }
            } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
#if defined(HAVE_SVE) && defined(HAVE_SVE2)
                if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128)) {
                    const size_t vlen_w = svcntw();
                    const size_t vlen_b = svcntb();
                    auto process_vec = [&](size_t lane_base, svbool_t pg_b8) {
                        svuint32_t v_u32[4];
                        svint32_t v_s32[4];
                        svint32_t best_idx[4];
                        svbool_t pg_w[4];

                        for (size_t s = 0; s < 4; ++s) {
                            const size_t lane_s = lane_base + s * vlen_w;
                            pg_w[s] = svwhilelt_b32(lane_s, work_amount);
                            const auto* base = src + lane_s;
                            if constexpr (std::is_same_v<T, int8_t>) {
                                v_s32[s] = svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base));
                                best_idx[s] = svdup_s32(0);
                            } else {
                                v_u32[s] = svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base));
                                best_idx[s] = svdup_s32(0);
                            }
                        }

                        const size_t stride = args->sort_stride;
                        const T* ptr_s[4];
                        for (size_t s = 0; s < 4; ++s) {
                            const size_t lane_s = lane_base + s * vlen_w;
                            ptr_s[s] = src + lane_s + stride;
                        }

                        if constexpr (std::is_same_v<T, int8_t>) {
                            auto update = [&](size_t s, svint32_t v, int32_t idx) {
                                const svbool_t mask =
                                    mode_max ? svcmpgt_s32(pg_w[s], v, v_s32[s])
                                             : svcmplt_s32(pg_w[s], v, v_s32[s]);
                                v_s32[s] = svsel(mask, v, v_s32[s]);
                                best_idx[s] = svsel(mask, svdup_s32(idx), best_idx[s]);
                            };

                            size_t i = 1;
                            for (; i + 3 < axis_dim; i += 4) {
                                for (size_t s = 0; s < 4; ++s) {
                                    const auto* base0 = ptr_s[s];
                                    const auto* base1 = ptr_s[s] + stride;
                                    const auto* base2 = ptr_s[s] + 2 * stride;
                                    const auto* base3 = ptr_s[s] + 3 * stride;
                                    update(s, svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base0)),
                                           static_cast<int32_t>(i + 0));
                                    update(s, svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base1)),
                                           static_cast<int32_t>(i + 1));
                                    update(s, svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base2)),
                                           static_cast<int32_t>(i + 2));
                                    update(s, svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base3)),
                                           static_cast<int32_t>(i + 3));
                                    ptr_s[s] += 4 * stride;
                                }
                            }
                            for (; i < axis_dim; ++i) {
                                for (size_t s = 0; s < 4; ++s) {
                                    const auto* base = ptr_s[s];
                                    update(s, svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base)),
                                           static_cast<int32_t>(i));
                                    ptr_s[s] += stride;
                                }
                            }
                        } else {
                            auto update = [&](size_t s, svuint32_t v, int32_t idx) {
                                const svbool_t mask =
                                    mode_max ? svcmpgt_u32(pg_w[s], v, v_u32[s])
                                             : svcmplt_u32(pg_w[s], v, v_u32[s]);
                                v_u32[s] = svsel(mask, v, v_u32[s]);
                                best_idx[s] = svsel(mask, svdup_s32(idx), best_idx[s]);
                            };

                            size_t i = 1;
                            for (; i + 3 < axis_dim; i += 4) {
                                for (size_t s = 0; s < 4; ++s) {
                                    const auto* base0 = ptr_s[s];
                                    const auto* base1 = ptr_s[s] + stride;
                                    const auto* base2 = ptr_s[s] + 2 * stride;
                                    const auto* base3 = ptr_s[s] + 3 * stride;
                                    update(s, svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base0)),
                                           static_cast<int32_t>(i + 0));
                                    update(s, svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base1)),
                                           static_cast<int32_t>(i + 1));
                                    update(s, svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base2)),
                                           static_cast<int32_t>(i + 2));
                                    update(s, svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base3)),
                                           static_cast<int32_t>(i + 3));
                                    ptr_s[s] += 4 * stride;
                                }
                            }
                            for (; i < axis_dim; ++i) {
                                for (size_t s = 0; s < 4; ++s) {
                                    const auto* base = ptr_s[s];
                                    update(s, svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base)),
                                           static_cast<int32_t>(i));
                                    ptr_s[s] += stride;
                                }
                            }
                        }

                        for (size_t s = 0; s < 4; ++s) {
                            const size_t lane_s = lane_base + s * vlen_w;
                            svst1(pg_w[s], dst_idx + lane_s, best_idx[s]);
                        }

                        if constexpr (std::is_same_v<T, int8_t>) {
                            const svint16_t pa = svqxtnt_s32(svqxtnb_s32(v_s32[0]), v_s32[1]);
                            const svint16_t pb = svqxtnt_s32(svqxtnb_s32(v_s32[2]), v_s32[3]);
                            const svint8_t res = svqxtnt_s16(svqxtnb_s16(pa), pb);
                            svst1(pg_b8, reinterpret_cast<int8_t*>(dst + lane_base), res);
                        } else {
                            const svuint16_t pa = svqxtnt_u32(svqxtnb_u32(v_u32[0]), v_u32[1]);
                            const svuint16_t pb = svqxtnt_u32(svqxtnb_u32(v_u32[2]), v_u32[3]);
                            const svuint8_t res = svqxtnt_u16(svqxtnb_u16(pa), pb);
                            svst1(pg_b8, reinterpret_cast<uint8_t*>(dst + lane_base), res);
                        }
                    };

                    for (; lane + vlen_b <= work_amount; lane += vlen_b) {
                        process_vec(lane, svptrue_b8());
                    }
                    if (lane < work_amount) {
                        process_vec(lane, svwhilelt_b8(lane, work_amount));
                    }
                    return;
                }
#endif  // HAVE_SVE && HAVE_SVE2
                constexpr size_t V = 16;
                for (; lane + V <= work_amount; lane += V) {
                    const auto* base = src + lane;
                    uint8x16_t best_u8;
                    if constexpr (std::is_same_v<T, int8_t>) {
                        best_u8 = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(base)));
                    } else {
                        best_u8 = vld1q_u8(reinterpret_cast<const uint8_t*>(base));
                    }

                    int32x4_t best_idx0 = vdupq_n_s32(0);
                    int32x4_t best_idx1 = vdupq_n_s32(0);
                    int32x4_t best_idx2 = vdupq_n_s32(0);
                    int32x4_t best_idx3 = vdupq_n_s32(0);
                    const size_t stride = args->sort_stride;
                    const T* ptr = src + stride + lane;

                    auto update = [&](uint8x16_t v_u8, int32_t idx) {
                        uint8x16_t mask8;
                        if constexpr (std::is_same_v<T, int8_t>) {
                            const int8x16_t v_s8 = vreinterpretq_s8_u8(v_u8);
                            const int8x16_t b_s8 = vreinterpretq_s8_u8(best_u8);
                            mask8 = mode_max ? vcgtq_s8(v_s8, b_s8) : vcltq_s8(v_s8, b_s8);
                        } else {
                            mask8 = mode_max ? vcgtq_u8(v_u8, best_u8) : vcltq_u8(v_u8, best_u8);
                        }

                        best_u8 = vbslq_u8(mask8, v_u8, best_u8);

                        const int32x4_t idx_vec = vdupq_n_s32(idx);
                        uint32x4_t m32_0, m32_1, m32_2, m32_3;
                        expand_mask_u8(mask8, m32_0, m32_1, m32_2, m32_3);
                        best_idx0 = vbslq_s32(m32_0, idx_vec, best_idx0);
                        best_idx1 = vbslq_s32(m32_1, idx_vec, best_idx1);
                        best_idx2 = vbslq_s32(m32_2, idx_vec, best_idx2);
                        best_idx3 = vbslq_s32(m32_3, idx_vec, best_idx3);
                    };

                    size_t i = 1;
                    for (; i + 3 < axis_dim; i += 4) {
                        const auto* ptr0 = ptr;
                        const auto* ptr1 = ptr + stride;
                        const auto* ptr2 = ptr + 2 * stride;
                        const auto* ptr3 = ptr + 3 * stride;
                        uint8x16_t v0, v1, v2, v3;
                        if constexpr (std::is_same_v<T, int8_t>) {
                            v0 = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(ptr0)));
                            v1 = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(ptr1)));
                            v2 = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(ptr2)));
                            v3 = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(ptr3)));
                        } else {
                            v0 = vld1q_u8(reinterpret_cast<const uint8_t*>(ptr0));
                            v1 = vld1q_u8(reinterpret_cast<const uint8_t*>(ptr1));
                            v2 = vld1q_u8(reinterpret_cast<const uint8_t*>(ptr2));
                            v3 = vld1q_u8(reinterpret_cast<const uint8_t*>(ptr3));
                        }
                        update(v0, static_cast<int32_t>(i + 0));
                        update(v1, static_cast<int32_t>(i + 1));
                        update(v2, static_cast<int32_t>(i + 2));
                        update(v3, static_cast<int32_t>(i + 3));
                        ptr += 4 * stride;
                    }
                    for (; i < axis_dim; ++i) {
                        uint8x16_t v_u8;
                        if constexpr (std::is_same_v<T, int8_t>) {
                            v_u8 = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(ptr)));
                        } else {
                            v_u8 = vld1q_u8(reinterpret_cast<const uint8_t*>(ptr));
                        }
                        update(v_u8, static_cast<int32_t>(i));
                        ptr += stride;
                    }

                    auto* out = dst + lane;
                    auto* out_idx = dst_idx + lane;
                    if constexpr (std::is_same_v<T, int8_t>) {
                        vst1q_s8(reinterpret_cast<int8_t*>(out), vreinterpretq_s8_u8(best_u8));
                    } else {
                        vst1q_u8(reinterpret_cast<uint8_t*>(out), best_u8);
                    }
                    vst1q_s32(out_idx + 0, best_idx0);
                    vst1q_s32(out_idx + 4, best_idx1);
                    vst1q_s32(out_idx + 8, best_idx2);
                    vst1q_s32(out_idx + 12, best_idx3);
                }
            }

            for (; lane < work_amount; ++lane) {
                T best = load_value(src, 0, lane, jcp, args);
                int32_t best_idx = 0;
                for (size_t i = 1; i < axis_dim; ++i) {
                    const T v = load_value(src, i, lane, jcp, args);
                    const bool better = mode_max ? TopkCompare<T>::gt(v, best) : TopkCompare<T>::lt(v, best);
                    if (better) {
                        best = v;
                        best_idx = static_cast<int32_t>(i);
                    }
                }
                store_value(dst, 0, lane, jcp, args, best);
                store_index(dst_idx, 0, lane, jcp, args, best_idx);
            }
        } else {
            for (size_t lane = 0; lane < work_amount; ++lane) {
                T best = load_value(src, 0, lane, jcp, args);
                int32_t best_idx = 0;
                for (size_t i = 1; i < axis_dim; ++i) {
                    const T v = load_value(src, i, lane, jcp, args);
                    const bool better = mode_max ? TopkCompare<T>::gt(v, best) : TopkCompare<T>::lt(v, best);
                    if (better) {
                        best = v;
                        best_idx = static_cast<int32_t>(i);
                    }
                }
                store_value(dst, 0, lane, jcp, args, best);
                store_index(dst_idx, 0, lane, jcp, args, best_idx);
            }
        }
        return;
    }

    constexpr size_t kStackMax = 64;
    TopkBuffers<T, kStackMax> buffers(top_k);
    T* top_vals = buffers.vals;
    int32_t* top_indices = buffers.idx;

    const bool mode_max = jcp->mode_max;

    if (!jcp->topk_innermost && !jcp->sort_index) {
        if constexpr (std::is_same_v<T, float>) {
#if defined(HAVE_SVE)
            if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128)) {
                constexpr size_t kVecMax = 8;
                if (top_k <= kVecMax) {
                    const size_t vlen = svcntw();
                    auto process_vec = [&](svbool_t pg, size_t lane_base) {
                        svfloat32_t vals[kVecMax];
                        svint32_t idx[kVecMax];
                        for (size_t k = 0; k < top_k; ++k) {
                            const auto* base =
                                reinterpret_cast<const float*>(src + k * args->sort_stride + lane_base);
                            vals[k] = svld1(pg, base);
                            idx[k] = svdup_s32(static_cast<int32_t>(k));
                        }

                        for (size_t i = 0; i + 1 < top_k; ++i) {
                            for (size_t j = top_k - 1; j > i; --j) {
                                const svbool_t mask =
                                    mode_max ? svcmpgt_f32(pg, vals[j], vals[j - 1])
                                             : svcmplt_f32(pg, vals[j], vals[j - 1]);
                                const svfloat32_t v_hi = svsel(mask, vals[j], vals[j - 1]);
                                const svfloat32_t v_lo = svsel(mask, vals[j - 1], vals[j]);
                                vals[j - 1] = v_hi;
                                vals[j] = v_lo;
                                const svint32_t i_hi = svsel(mask, idx[j], idx[j - 1]);
                                const svint32_t i_lo = svsel(mask, idx[j - 1], idx[j]);
                                idx[j - 1] = i_hi;
                                idx[j] = i_lo;
                            }
                        }

                        auto insert_val = [&](svfloat32_t v, int32_t v_i) {
                            svint32_t v_idx = svdup_s32(v_i);
                            for (size_t k = 0; k < top_k; ++k) {
                                const svbool_t mask =
                                    mode_max ? svcmpgt_f32(pg, v, vals[k]) : svcmplt_f32(pg, v, vals[k]);
                                const svfloat32_t v_new = svsel(mask, vals[k], v);
                                const svfloat32_t slot_new = svsel(mask, v, vals[k]);
                                v = v_new;
                                vals[k] = slot_new;
                                const svint32_t i_new = svsel(mask, idx[k], v_idx);
                                const svint32_t slot_i = svsel(mask, v_idx, idx[k]);
                                v_idx = i_new;
                                idx[k] = slot_i;
                            }
                        };

                        size_t i = top_k;
                        const size_t stride = args->sort_stride;
                        const T* ptr = src + lane_base + top_k * stride;
                        for (; i + 1 < axis_dim; i += 2) {
                            const auto* base0 = reinterpret_cast<const float*>(ptr);
                            const auto* base1 = reinterpret_cast<const float*>(ptr + stride);
                            insert_val(svld1(pg, base0), static_cast<int32_t>(i));
                            insert_val(svld1(pg, base1), static_cast<int32_t>(i + 1));
                            ptr += 2 * stride;
                        }
                        for (; i < axis_dim; ++i) {
                            const auto* base = reinterpret_cast<const float*>(ptr);
                            insert_val(svld1(pg, base), static_cast<int32_t>(i));
                            ptr += stride;
                        }

                        for (size_t k = 0; k < top_k; ++k) {
                            auto* out = reinterpret_cast<float*>(dst + k * args->sort_stride + lane_base);
                            auto* out_idx = dst_idx + k * args->sort_stride + lane_base;
                            svst1(pg, out, vals[k]);
                            svst1(pg, out_idx, idx[k]);
                        }
                    };

                    size_t lane = 0;
                    for (; lane + vlen <= work_amount; lane += vlen) {
                        process_vec(svptrue_b32(), lane);
                    }
                    if (lane < work_amount) {
                        process_vec(svwhilelt_b32(lane, work_amount), lane);
                    }
                    return;
                }
            }
#endif
            constexpr size_t V = 4;
            constexpr size_t kVecMax = 8;
            if (top_k <= kVecMax && work_amount >= V) {
                size_t lane = 0;
                for (; lane + V <= work_amount; lane += V) {
                    float32x4_t vals[kVecMax];
                    int32x4_t idx[kVecMax];
                    for (size_t k = 0; k < top_k; ++k) {
                        vals[k] = vld1q_f32(reinterpret_cast<const float*>(src + k * args->sort_stride + lane));
                        idx[k] = vdupq_n_s32(static_cast<int32_t>(k));
                    }

                    for (size_t i = 0; i + 1 < top_k; ++i) {
                        for (size_t j = top_k - 1; j > i; --j) {
                            const uint32x4_t mask =
                                mode_max ? vcgtq_f32(vals[j], vals[j - 1]) : vcltq_f32(vals[j], vals[j - 1]);
                            const float32x4_t v_hi = vbslq_f32(mask, vals[j], vals[j - 1]);
                            const float32x4_t v_lo = vbslq_f32(mask, vals[j - 1], vals[j]);
                            vals[j - 1] = v_hi;
                            vals[j] = v_lo;
                            const int32x4_t i_hi = vbslq_s32(mask, idx[j], idx[j - 1]);
                            const int32x4_t i_lo = vbslq_s32(mask, idx[j - 1], idx[j]);
                            idx[j - 1] = i_hi;
                            idx[j] = i_lo;
                        }
                    }

                    auto insert_val = [&](float32x4_t v, int32_t v_i) {
                        int32x4_t v_idx = vdupq_n_s32(v_i);
                        for (size_t k = 0; k < top_k; ++k) {
                            const uint32x4_t mask =
                                mode_max ? vcgtq_f32(v, vals[k]) : vcltq_f32(v, vals[k]);
                            const float32x4_t v_new = vbslq_f32(mask, vals[k], v);
                            const float32x4_t slot_new = vbslq_f32(mask, v, vals[k]);
                            v = v_new;
                            vals[k] = slot_new;
                            const int32x4_t i_new = vbslq_s32(mask, idx[k], v_idx);
                            const int32x4_t slot_i = vbslq_s32(mask, v_idx, idx[k]);
                            v_idx = i_new;
                            idx[k] = slot_i;
                        }
                    };

                    size_t i = top_k;
                    const size_t stride = args->sort_stride;
                    const T* ptr = src + lane + top_k * stride;
                    for (; i + 1 < axis_dim; i += 2) {
                        insert_val(vld1q_f32(reinterpret_cast<const float*>(ptr)),
                                   static_cast<int32_t>(i));
                        insert_val(vld1q_f32(reinterpret_cast<const float*>(ptr + stride)),
                                   static_cast<int32_t>(i + 1));
                        ptr += 2 * stride;
                    }
                    for (; i < axis_dim; ++i) {
                        insert_val(vld1q_f32(reinterpret_cast<const float*>(ptr)),
                                   static_cast<int32_t>(i));
                        ptr += stride;
                    }

                    for (size_t k = 0; k < top_k; ++k) {
                        auto* out = dst + k * args->sort_stride + lane;
                        auto* out_idx = dst_idx + k * args->sort_stride + lane;
                        vst1q_f32(reinterpret_cast<float*>(out), vals[k]);
                        vst1q_s32(out_idx, idx[k]);
                    }
                }

                for (; lane < work_amount; ++lane) {
                    for (size_t i = 0; i < top_k; ++i) {
                        top_vals[i] = load_value(src, i, lane, jcp, args);
                        top_indices[i] = static_cast<int32_t>(i);
                    }

                    for (size_t i = 0; i + 1 < top_k; ++i) {
                        for (size_t j = top_k - 1; j > i; --j) {
                            const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                                      : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                            if (cmp) {
                                std::swap(top_vals[j], top_vals[j - 1]);
                                std::swap(top_indices[j], top_indices[j - 1]);
                            }
                        }
                    }

                    for (size_t i = top_k; i < axis_dim; ++i) {
                        top_vals[top_k] = load_value(src, i, lane, jcp, args);
                        top_indices[top_k] = static_cast<int32_t>(i);
                        for (size_t j = top_k; j > 0; --j) {
                            const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                                      : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                            if (cmp) {
                                std::swap(top_vals[j], top_vals[j - 1]);
                                std::swap(top_indices[j], top_indices[j - 1]);
                            } else {
                                break;
                            }
                        }
                    }

                    for (size_t i = 0; i < top_k; ++i) {
                        store_value(dst, i, lane, jcp, args, top_vals[i]);
                        store_index(dst_idx, i, lane, jcp, args, top_indices[i]);
                    }
                }
                return;
            }
        } else if constexpr (std::is_same_v<T, ov::float16>) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            size_t lane = 0;
#    if defined(HAVE_SVE)
            if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128)) {
                constexpr size_t kVecMax = 8;
                if (top_k <= kVecMax) {
                    const size_t vlen_h = svcnth();
                    const size_t vlen_f32 = svcntw();
                    const svbool_t pg_f16 = svptrue_b16();
                    const svbool_t pg_f32 = svptrue_b32();
                    for (; lane + vlen_h <= work_amount; lane += vlen_h) {
                        svfloat32_t vals_low[kVecMax];
                        svfloat32_t vals_high[kVecMax];
                        svint32_t idx_low[kVecMax];
                        svint32_t idx_high[kVecMax];
                        for (size_t k = 0; k < top_k; ++k) {
                            const auto* base =
                                reinterpret_cast<const float16_t*>(src + k * args->sort_stride + lane);
                            svfloat16_t v_f16 = svld1(pg_f16, base);
                            vals_low[k] = sve_cvt_f32_f16_low(pg_f32, v_f16);
                            vals_high[k] = sve_cvt_f32_f16_high(pg_f32, v_f16);
                            idx_low[k] = svdup_s32(static_cast<int32_t>(k));
                            idx_high[k] = svdup_s32(static_cast<int32_t>(k));
                        }

                        for (size_t i = 0; i + 1 < top_k; ++i) {
                            for (size_t j = top_k - 1; j > i; --j) {
                                const svbool_t mask_low =
                                    mode_max ? svcmpgt_f32(pg_f32, vals_low[j], vals_low[j - 1])
                                             : svcmplt_f32(pg_f32, vals_low[j], vals_low[j - 1]);
                                const svbool_t mask_high =
                                    mode_max ? svcmpgt_f32(pg_f32, vals_high[j], vals_high[j - 1])
                                             : svcmplt_f32(pg_f32, vals_high[j], vals_high[j - 1]);
                                const svfloat32_t v_hi_low = svsel(mask_low, vals_low[j], vals_low[j - 1]);
                                const svfloat32_t v_lo_low = svsel(mask_low, vals_low[j - 1], vals_low[j]);
                                const svfloat32_t v_hi_high = svsel(mask_high, vals_high[j], vals_high[j - 1]);
                                const svfloat32_t v_lo_high = svsel(mask_high, vals_high[j - 1], vals_high[j]);
                                vals_low[j - 1] = v_hi_low;
                                vals_low[j] = v_lo_low;
                                vals_high[j - 1] = v_hi_high;
                                vals_high[j] = v_lo_high;
                                const svint32_t i_hi_low = svsel(mask_low, idx_low[j], idx_low[j - 1]);
                                const svint32_t i_lo_low = svsel(mask_low, idx_low[j - 1], idx_low[j]);
                                const svint32_t i_hi_high = svsel(mask_high, idx_high[j], idx_high[j - 1]);
                                const svint32_t i_lo_high = svsel(mask_high, idx_high[j - 1], idx_high[j]);
                                idx_low[j - 1] = i_hi_low;
                                idx_low[j] = i_lo_low;
                                idx_high[j - 1] = i_hi_high;
                                idx_high[j] = i_lo_high;
                            }
                        }

                        auto insert_val = [&](svfloat16_t v_f16, int32_t v_i) {
                            svfloat32_t v_low = sve_cvt_f32_f16_low(pg_f32, v_f16);
                            svfloat32_t v_high = sve_cvt_f32_f16_high(pg_f32, v_f16);
                            svint32_t v_idx = svdup_s32(v_i);
                            svint32_t v_idx_high = v_idx;
                            for (size_t k = 0; k < top_k; ++k) {
                                const svbool_t mask_low =
                                    mode_max ? svcmpgt_f32(pg_f32, v_low, vals_low[k])
                                             : svcmplt_f32(pg_f32, v_low, vals_low[k]);
                                const svbool_t mask_high =
                                    mode_max ? svcmpgt_f32(pg_f32, v_high, vals_high[k])
                                             : svcmplt_f32(pg_f32, v_high, vals_high[k]);
                                const svfloat32_t v_new_low = svsel(mask_low, vals_low[k], v_low);
                                const svfloat32_t slot_low = svsel(mask_low, v_low, vals_low[k]);
                                const svfloat32_t v_new_high = svsel(mask_high, vals_high[k], v_high);
                                const svfloat32_t slot_high = svsel(mask_high, v_high, vals_high[k]);
                                v_low = v_new_low;
                                v_high = v_new_high;
                                vals_low[k] = slot_low;
                                vals_high[k] = slot_high;

                                const svint32_t i_new_low = svsel(mask_low, idx_low[k], v_idx);
                                const svint32_t slot_i_low = svsel(mask_low, v_idx, idx_low[k]);
                                const svint32_t i_new_high = svsel(mask_high, idx_high[k], v_idx_high);
                                const svint32_t slot_i_high = svsel(mask_high, v_idx_high, idx_high[k]);
                                v_idx = i_new_low;
                                v_idx_high = i_new_high;
                                idx_low[k] = slot_i_low;
                                idx_high[k] = slot_i_high;
                            }
                        };

                        size_t i = top_k;
                        const size_t stride = args->sort_stride;
                        const T* ptr = src + lane + top_k * stride;
                        for (; i + 1 < axis_dim; i += 2) {
                            const auto* base0 = reinterpret_cast<const float16_t*>(ptr);
                            const auto* base1 = reinterpret_cast<const float16_t*>(ptr + stride);
                            insert_val(svld1(pg_f16, base0), static_cast<int32_t>(i));
                            insert_val(svld1(pg_f16, base1), static_cast<int32_t>(i + 1));
                            ptr += 2 * stride;
                        }
                        for (; i < axis_dim; ++i) {
                            const auto* base = reinterpret_cast<const float16_t*>(ptr);
                            insert_val(svld1(pg_f16, base), static_cast<int32_t>(i));
                            ptr += stride;
                        }

                        for (size_t k = 0; k < top_k; ++k) {
                            auto* out = reinterpret_cast<float16_t*>(dst + k * args->sort_stride + lane);
                            auto* out_idx = dst_idx + k * args->sort_stride + lane;
                            svfloat16_t out_f16 = sve_pack_f16_from_f32(pg_f32, vals_low[k], vals_high[k]);
                            svst1(pg_f16, out, out_f16);
                            svst1(pg_f32, out_idx, idx_low[k]);
                            svst1(pg_f32, out_idx + vlen_f32, idx_high[k]);
                        }
                    }

                    if (lane == work_amount) {
                        return;
                    }
                    // NEON/scalar will handle the tail
                }
            }
#    endif  // HAVE_SVE
            constexpr size_t V = 8;
            constexpr size_t kVecMax = 8;
            if (top_k <= kVecMax && work_amount >= V) {
                for (; lane + V <= work_amount; lane += V) {
                    float32x4_t vals_low[kVecMax];
                    float32x4_t vals_high[kVecMax];
                    int32x4_t idx0[kVecMax];
                    int32x4_t idx1[kVecMax];
                    for (size_t k = 0; k < top_k; ++k) {
                        float16x8_t v = load_f16x8(src + k * args->sort_stride + lane);
                        vals_low[k] = vcvt_f32_f16(vget_low_f16(v));
                        vals_high[k] = vcvt_f32_f16(vget_high_f16(v));
                        idx0[k] = vdupq_n_s32(static_cast<int32_t>(k));
                        idx1[k] = vdupq_n_s32(static_cast<int32_t>(k));
                    }

                    for (size_t i = 0; i + 1 < top_k; ++i) {
                        for (size_t j = top_k - 1; j > i; --j) {
                            const uint32x4_t mask_low =
                                mode_max ? vcgtq_f32(vals_low[j], vals_low[j - 1]) : vcltq_f32(vals_low[j], vals_low[j - 1]);
                            const uint32x4_t mask_high =
                                mode_max ? vcgtq_f32(vals_high[j], vals_high[j - 1]) : vcltq_f32(vals_high[j], vals_high[j - 1]);
                            const float32x4_t v_hi_low = vbslq_f32(mask_low, vals_low[j], vals_low[j - 1]);
                            const float32x4_t v_lo_low = vbslq_f32(mask_low, vals_low[j - 1], vals_low[j]);
                            const float32x4_t v_hi_high = vbslq_f32(mask_high, vals_high[j], vals_high[j - 1]);
                            const float32x4_t v_lo_high = vbslq_f32(mask_high, vals_high[j - 1], vals_high[j]);
                            vals_low[j - 1] = v_hi_low;
                            vals_low[j] = v_lo_low;
                            vals_high[j - 1] = v_hi_high;
                            vals_high[j] = v_lo_high;
                            const int32x4_t i0_hi = vbslq_s32(mask_low, idx0[j], idx0[j - 1]);
                            const int32x4_t i0_lo = vbslq_s32(mask_low, idx0[j - 1], idx0[j]);
                            const int32x4_t i1_hi = vbslq_s32(mask_high, idx1[j], idx1[j - 1]);
                            const int32x4_t i1_lo = vbslq_s32(mask_high, idx1[j - 1], idx1[j]);
                            idx0[j - 1] = i0_hi;
                            idx0[j] = i0_lo;
                            idx1[j - 1] = i1_hi;
                            idx1[j] = i1_lo;
                        }
                    }

                    auto insert_val = [&](float16x8_t v, int32_t v_i) {
                        float32x4_t v_low = vcvt_f32_f16(vget_low_f16(v));
                        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v));
                        int32x4_t v_idx0 = vdupq_n_s32(v_i);
                        int32x4_t v_idx1 = vdupq_n_s32(v_i);
                        for (size_t k = 0; k < top_k; ++k) {
                            const uint32x4_t cmp_low =
                                mode_max ? vcgtq_f32(v_low, vals_low[k]) : vcltq_f32(v_low, vals_low[k]);
                            const uint32x4_t cmp_high =
                                mode_max ? vcgtq_f32(v_high, vals_high[k]) : vcltq_f32(v_high, vals_high[k]);
                            const uint32x4_t eq_low = vceqq_f32(v_low, vals_low[k]);
                            const uint32x4_t eq_high = vceqq_f32(v_high, vals_high[k]);
                            const uint32x4_t idx_low = vcltq_s32(v_idx0, idx0[k]);
                            const uint32x4_t idx_high = vcltq_s32(v_idx1, idx1[k]);
                            const uint32x4_t mask_low = vorrq_u32(cmp_low, vandq_u32(eq_low, idx_low));
                            const uint32x4_t mask_high = vorrq_u32(cmp_high, vandq_u32(eq_high, idx_high));
                            const float32x4_t v_new_low = vbslq_f32(mask_low, vals_low[k], v_low);
                            const float32x4_t slot_low = vbslq_f32(mask_low, v_low, vals_low[k]);
                            const float32x4_t v_new_high = vbslq_f32(mask_high, vals_high[k], v_high);
                            const float32x4_t slot_high = vbslq_f32(mask_high, v_high, vals_high[k]);
                            v_low = v_new_low;
                            v_high = v_new_high;
                            vals_low[k] = slot_low;
                            vals_high[k] = slot_high;

                            const int32x4_t i_new0 = vbslq_s32(mask_low, idx0[k], v_idx0);
                            const int32x4_t i_new1 = vbslq_s32(mask_high, idx1[k], v_idx1);
                            const int32x4_t slot_i0 = vbslq_s32(mask_low, v_idx0, idx0[k]);
                            const int32x4_t slot_i1 = vbslq_s32(mask_high, v_idx1, idx1[k]);
                            v_idx0 = i_new0;
                            v_idx1 = i_new1;
                            idx0[k] = slot_i0;
                            idx1[k] = slot_i1;
                        }
                    };

                    size_t i = top_k;
                    const size_t stride = args->sort_stride;
                    const T* ptr = src + lane + top_k * stride;
                    for (; i + 1 < axis_dim; i += 2) {
                        insert_val(load_f16x8(ptr),
                                   static_cast<int32_t>(i));
                        insert_val(load_f16x8(ptr + stride),
                                   static_cast<int32_t>(i + 1));
                        ptr += 2 * stride;
                    }
                    for (; i < axis_dim; ++i) {
                        insert_val(load_f16x8(ptr),
                                   static_cast<int32_t>(i));
                        ptr += stride;
                    }

                    for (size_t k = 0; k < top_k; ++k) {
                        auto* out = dst + k * args->sort_stride + lane;
                        auto* out_idx = dst_idx + k * args->sort_stride + lane;
                        const float16x4_t out_low = vcvt_f16_f32(vals_low[k]);
                        const float16x4_t out_high = vcvt_f16_f32(vals_high[k]);
                        store_f16x8(out, vcombine_f16(out_low, out_high));
                        vst1q_s32(out_idx + 0, idx0[k]);
                        vst1q_s32(out_idx + 4, idx1[k]);
                    }
                }

                for (; lane < work_amount; ++lane) {
                    for (size_t i = 0; i < top_k; ++i) {
                        top_vals[i] = load_value(src, i, lane, jcp, args);
                        top_indices[i] = static_cast<int32_t>(i);
                    }

                    for (size_t i = 0; i + 1 < top_k; ++i) {
                        for (size_t j = top_k - 1; j > i; --j) {
                            const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                                      : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                            if (cmp) {
                                std::swap(top_vals[j], top_vals[j - 1]);
                                std::swap(top_indices[j], top_indices[j - 1]);
                            }
                        }
                    }

                    for (size_t i = top_k; i < axis_dim; ++i) {
                        top_vals[top_k] = load_value(src, i, lane, jcp, args);
                        top_indices[top_k] = static_cast<int32_t>(i);
                        for (size_t j = top_k; j > 0; --j) {
                            const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                                      : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                            if (cmp) {
                                std::swap(top_vals[j], top_vals[j - 1]);
                                std::swap(top_indices[j], top_indices[j - 1]);
                            } else {
                                break;
                            }
                        }
                    }

                    for (size_t i = 0; i < top_k; ++i) {
                        store_value(dst, i, lane, jcp, args, top_vals[i]);
                        store_index(dst_idx, i, lane, jcp, args, top_indices[i]);
                    }
                }
                return;
            }
#endif
        } else if constexpr (std::is_same_v<T, int32_t>) {
#if defined(HAVE_SVE)
            if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128)) {
                constexpr size_t kVecMax = 8;
                if (top_k <= kVecMax) {
                    const size_t vlen = svcntw();
                    auto process_vec = [&](svbool_t pg, size_t lane_base) {
                        svint32_t vals[kVecMax];
                        svint32_t idx[kVecMax];
                        for (size_t k = 0; k < top_k; ++k) {
                            const auto* base =
                                reinterpret_cast<const int32_t*>(src + k * args->sort_stride + lane_base);
                            vals[k] = svld1(pg, base);
                            idx[k] = svdup_s32(static_cast<int32_t>(k));
                        }

                        for (size_t i = 0; i + 1 < top_k; ++i) {
                            for (size_t j = top_k - 1; j > i; --j) {
                                const svbool_t mask =
                                    mode_max ? svcmpgt_s32(pg, vals[j], vals[j - 1])
                                             : svcmplt_s32(pg, vals[j], vals[j - 1]);
                                const svint32_t v_hi = svsel(mask, vals[j], vals[j - 1]);
                                const svint32_t v_lo = svsel(mask, vals[j - 1], vals[j]);
                                vals[j - 1] = v_hi;
                                vals[j] = v_lo;
                                const svint32_t i_hi = svsel(mask, idx[j], idx[j - 1]);
                                const svint32_t i_lo = svsel(mask, idx[j - 1], idx[j]);
                                idx[j - 1] = i_hi;
                                idx[j] = i_lo;
                            }
                        }

                        auto insert_val = [&](svint32_t v, int32_t v_i) {
                            svint32_t v_idx = svdup_s32(v_i);
                            for (size_t k = 0; k < top_k; ++k) {
                                const svbool_t mask =
                                    mode_max ? svcmpgt_s32(pg, v, vals[k]) : svcmplt_s32(pg, v, vals[k]);
                                const svint32_t v_new = svsel(mask, vals[k], v);
                                const svint32_t slot_new = svsel(mask, v, vals[k]);
                                v = v_new;
                                vals[k] = slot_new;
                                const svint32_t i_new = svsel(mask, idx[k], v_idx);
                                const svint32_t slot_i = svsel(mask, v_idx, idx[k]);
                                v_idx = i_new;
                                idx[k] = slot_i;
                            }
                        };

                        size_t i = top_k;
                        const size_t stride = args->sort_stride;
                        const T* ptr = src + lane_base + top_k * stride;
                        for (; i + 1 < axis_dim; i += 2) {
                            const auto* base0 = reinterpret_cast<const int32_t*>(ptr);
                            const auto* base1 = reinterpret_cast<const int32_t*>(ptr + stride);
                            insert_val(svld1(pg, base0), static_cast<int32_t>(i));
                            insert_val(svld1(pg, base1), static_cast<int32_t>(i + 1));
                            ptr += 2 * stride;
                        }
                        for (; i < axis_dim; ++i) {
                            const auto* base = reinterpret_cast<const int32_t*>(ptr);
                            insert_val(svld1(pg, base), static_cast<int32_t>(i));
                            ptr += stride;
                        }

                        for (size_t k = 0; k < top_k; ++k) {
                            auto* out = reinterpret_cast<int32_t*>(dst + k * args->sort_stride + lane_base);
                            auto* out_idx = dst_idx + k * args->sort_stride + lane_base;
                            svst1(pg, out, vals[k]);
                            svst1(pg, out_idx, idx[k]);
                        }
                    };

                    size_t lane = 0;
                    for (; lane + vlen <= work_amount; lane += vlen) {
                        process_vec(svptrue_b32(), lane);
                    }
                    if (lane < work_amount) {
                        process_vec(svwhilelt_b32(lane, work_amount), lane);
                    }
                    return;
                }
            }
#endif
            constexpr size_t V = 4;
            constexpr size_t kVecMax = 8;
            if (top_k <= kVecMax && work_amount >= V) {
                size_t lane = 0;
                for (; lane + V <= work_amount; lane += V) {
                    int32x4_t vals[kVecMax];
                    int32x4_t idx[kVecMax];
                    for (size_t k = 0; k < top_k; ++k) {
                        vals[k] = vld1q_s32(reinterpret_cast<const int32_t*>(src + k * args->sort_stride + lane));
                        idx[k] = vdupq_n_s32(static_cast<int32_t>(k));
                    }

                    for (size_t i = 0; i + 1 < top_k; ++i) {
                        for (size_t j = top_k - 1; j > i; --j) {
                            const uint32x4_t mask =
                                mode_max ? vcgtq_s32(vals[j], vals[j - 1]) : vcltq_s32(vals[j], vals[j - 1]);
                            const int32x4_t v_hi = vbslq_s32(mask, vals[j], vals[j - 1]);
                            const int32x4_t v_lo = vbslq_s32(mask, vals[j - 1], vals[j]);
                            vals[j - 1] = v_hi;
                            vals[j] = v_lo;
                            const int32x4_t i_hi = vbslq_s32(mask, idx[j], idx[j - 1]);
                            const int32x4_t i_lo = vbslq_s32(mask, idx[j - 1], idx[j]);
                            idx[j - 1] = i_hi;
                            idx[j] = i_lo;
                        }
                    }

                    auto insert_val = [&](int32x4_t v, int32_t v_i) {
                        int32x4_t v_idx = vdupq_n_s32(v_i);
                        for (size_t k = 0; k < top_k; ++k) {
                            const uint32x4_t mask =
                                mode_max ? vcgtq_s32(v, vals[k]) : vcltq_s32(v, vals[k]);
                            const int32x4_t v_new = vbslq_s32(mask, vals[k], v);
                            const int32x4_t slot_new = vbslq_s32(mask, v, vals[k]);
                            v = v_new;
                            vals[k] = slot_new;
                            const int32x4_t i_new = vbslq_s32(mask, idx[k], v_idx);
                            const int32x4_t slot_i = vbslq_s32(mask, v_idx, idx[k]);
                            v_idx = i_new;
                            idx[k] = slot_i;
                        }
                    };

                    size_t i = top_k;
                    const size_t stride = args->sort_stride;
                    const T* ptr = src + lane + top_k * stride;
                    for (; i + 1 < axis_dim; i += 2) {
                        insert_val(vld1q_s32(reinterpret_cast<const int32_t*>(ptr)), static_cast<int32_t>(i));
                        insert_val(vld1q_s32(reinterpret_cast<const int32_t*>(ptr + stride)),
                                   static_cast<int32_t>(i + 1));
                        ptr += 2 * stride;
                    }
                    for (; i < axis_dim; ++i) {
                        insert_val(vld1q_s32(reinterpret_cast<const int32_t*>(ptr)),
                                   static_cast<int32_t>(i));
                        ptr += stride;
                    }

                    for (size_t k = 0; k < top_k; ++k) {
                        auto* out = dst + k * args->sort_stride + lane;
                        auto* out_idx = dst_idx + k * args->sort_stride + lane;
                        vst1q_s32(reinterpret_cast<int32_t*>(out), vals[k]);
                        vst1q_s32(out_idx, idx[k]);
                    }
                }

                for (; lane < work_amount; ++lane) {
                    for (size_t i = 0; i < top_k; ++i) {
                        top_vals[i] = load_value(src, i, lane, jcp, args);
                        top_indices[i] = static_cast<int32_t>(i);
                    }

                    for (size_t i = 0; i + 1 < top_k; ++i) {
                        for (size_t j = top_k - 1; j > i; --j) {
                            const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                                      : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                            if (cmp) {
                                std::swap(top_vals[j], top_vals[j - 1]);
                                std::swap(top_indices[j], top_indices[j - 1]);
                            }
                        }
                    }

                    for (size_t i = top_k; i < axis_dim; ++i) {
                        top_vals[top_k] = load_value(src, i, lane, jcp, args);
                        top_indices[top_k] = static_cast<int32_t>(i);
                        for (size_t j = top_k; j > 0; --j) {
                            const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                                      : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                            if (cmp) {
                                std::swap(top_vals[j], top_vals[j - 1]);
                                std::swap(top_indices[j], top_indices[j - 1]);
                            } else {
                                break;
                            }
                        }
                    }

                    for (size_t i = 0; i < top_k; ++i) {
                        store_value(dst, i, lane, jcp, args, top_vals[i]);
                        store_index(dst_idx, i, lane, jcp, args, top_indices[i]);
                    }
                }
                return;
            }
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
#if defined(HAVE_SVE) && defined(HAVE_SVE2)
            if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128) &&
                ov::with_cpu_sve2()) {
                constexpr size_t kVecMax = 8;
                if (top_k <= kVecMax) {
                    const size_t vlen_w = svcntw();
                    const size_t vlen_b = svcntb();
                    auto process_vec = [&](size_t lane_base, svbool_t pg_b8) {
                        svbool_t pg_w[4];
                        svuint32_t vals_u32[kVecMax][4];
                        svint32_t vals_s32[kVecMax][4];
                        svint32_t idx[kVecMax][4];

                        for (size_t s = 0; s < 4; ++s) {
                            const size_t lane_s = lane_base + s * vlen_w;
                            pg_w[s] = svwhilelt_b32(lane_s, work_amount);
                        }

                        for (size_t k = 0; k < top_k; ++k) {
                            for (size_t s = 0; s < 4; ++s) {
                                const size_t lane_s = lane_base + s * vlen_w;
                                const auto* base = src + k * args->sort_stride + lane_s;
                                if constexpr (std::is_same_v<T, int8_t>) {
                                    vals_s32[k][s] = svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base));
                                } else {
                                    vals_u32[k][s] = svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base));
                                }
                                idx[k][s] = svdup_s32(static_cast<int32_t>(k));
                            }
                        }

                        for (size_t i = 0; i + 1 < top_k; ++i) {
                            for (size_t j = top_k - 1; j > i; --j) {
                                for (size_t s = 0; s < 4; ++s) {
                                    if constexpr (std::is_same_v<T, int8_t>) {
                                        const svbool_t mask =
                                            mode_max ? svcmpgt_s32(pg_w[s], vals_s32[j][s], vals_s32[j - 1][s])
                                                     : svcmplt_s32(pg_w[s], vals_s32[j][s], vals_s32[j - 1][s]);
                                        const svint32_t v_hi = svsel(mask, vals_s32[j][s], vals_s32[j - 1][s]);
                                        const svint32_t v_lo = svsel(mask, vals_s32[j - 1][s], vals_s32[j][s]);
                                        vals_s32[j - 1][s] = v_hi;
                                        vals_s32[j][s] = v_lo;
                                        const svint32_t i_hi = svsel(mask, idx[j][s], idx[j - 1][s]);
                                        const svint32_t i_lo = svsel(mask, idx[j - 1][s], idx[j][s]);
                                        idx[j - 1][s] = i_hi;
                                        idx[j][s] = i_lo;
                                    } else {
                                        const svbool_t mask =
                                            mode_max ? svcmpgt_u32(pg_w[s], vals_u32[j][s], vals_u32[j - 1][s])
                                                     : svcmplt_u32(pg_w[s], vals_u32[j][s], vals_u32[j - 1][s]);
                                        const svuint32_t v_hi = svsel(mask, vals_u32[j][s], vals_u32[j - 1][s]);
                                        const svuint32_t v_lo = svsel(mask, vals_u32[j - 1][s], vals_u32[j][s]);
                                        vals_u32[j - 1][s] = v_hi;
                                        vals_u32[j][s] = v_lo;
                                        const svint32_t i_hi = svsel(mask, idx[j][s], idx[j - 1][s]);
                                        const svint32_t i_lo = svsel(mask, idx[j - 1][s], idx[j][s]);
                                        idx[j - 1][s] = i_hi;
                                        idx[j][s] = i_lo;
                                    }
                                }
                            }
                        }

                        if constexpr (std::is_same_v<T, int8_t>) {
                            auto insert = [&](size_t s, svint32_t v, int32_t v_i) {
                                svint32_t v_idx = svdup_s32(v_i);
                                for (size_t k = 0; k < top_k; ++k) {
                                    const svbool_t mask =
                                        mode_max ? svcmpgt_s32(pg_w[s], v, vals_s32[k][s])
                                                 : svcmplt_s32(pg_w[s], v, vals_s32[k][s]);
                                    const svint32_t v_new = svsel(mask, vals_s32[k][s], v);
                                    const svint32_t slot_new = svsel(mask, v, vals_s32[k][s]);
                                    v = v_new;
                                    vals_s32[k][s] = slot_new;
                                    const svint32_t i_new = svsel(mask, idx[k][s], v_idx);
                                    const svint32_t slot_i = svsel(mask, v_idx, idx[k][s]);
                                    v_idx = i_new;
                                    idx[k][s] = slot_i;
                                }
                            };

                            const size_t stride = args->sort_stride;
                            const T* ptr_s[4];
                            for (size_t s = 0; s < 4; ++s) {
                                const size_t lane_s = lane_base + s * vlen_w;
                                ptr_s[s] = src + lane_s + top_k * stride;
                            }

                            size_t i = top_k;
                            for (; i + 1 < axis_dim; i += 2) {
                                for (size_t s = 0; s < 4; ++s) {
                                    const auto* base0 = ptr_s[s];
                                    const auto* base1 = ptr_s[s] + stride;
                                    insert(s, svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base0)),
                                           static_cast<int32_t>(i));
                                    insert(s, svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base1)),
                                           static_cast<int32_t>(i + 1));
                                    ptr_s[s] += 2 * stride;
                                }
                            }
                            for (; i < axis_dim; ++i) {
                                for (size_t s = 0; s < 4; ++s) {
                                    const auto* base = ptr_s[s];
                                    insert(s, svld1sb_s32(pg_w[s], reinterpret_cast<const int8_t*>(base)),
                                           static_cast<int32_t>(i));
                                    ptr_s[s] += stride;
                                }
                            }
                        } else {
                            auto insert = [&](size_t s, svuint32_t v, int32_t v_i) {
                                svint32_t v_idx = svdup_s32(v_i);
                                for (size_t k = 0; k < top_k; ++k) {
                                    const svbool_t mask =
                                        mode_max ? svcmpgt_u32(pg_w[s], v, vals_u32[k][s])
                                                 : svcmplt_u32(pg_w[s], v, vals_u32[k][s]);
                                    const svuint32_t v_new = svsel(mask, vals_u32[k][s], v);
                                    const svuint32_t slot_new = svsel(mask, v, vals_u32[k][s]);
                                    v = v_new;
                                    vals_u32[k][s] = slot_new;
                                    const svint32_t i_new = svsel(mask, idx[k][s], v_idx);
                                    const svint32_t slot_i = svsel(mask, v_idx, idx[k][s]);
                                    v_idx = i_new;
                                    idx[k][s] = slot_i;
                                }
                            };

                            const size_t stride = args->sort_stride;
                            const T* ptr_s[4];
                            for (size_t s = 0; s < 4; ++s) {
                                const size_t lane_s = lane_base + s * vlen_w;
                                ptr_s[s] = src + lane_s + top_k * stride;
                            }

                            size_t i = top_k;
                            for (; i + 1 < axis_dim; i += 2) {
                                for (size_t s = 0; s < 4; ++s) {
                                    const auto* base0 = ptr_s[s];
                                    const auto* base1 = ptr_s[s] + stride;
                                    insert(s, svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base0)),
                                           static_cast<int32_t>(i));
                                    insert(s, svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base1)),
                                           static_cast<int32_t>(i + 1));
                                    ptr_s[s] += 2 * stride;
                                }
                            }
                            for (; i < axis_dim; ++i) {
                                for (size_t s = 0; s < 4; ++s) {
                                    const auto* base = ptr_s[s];
                                    insert(s, svld1ub_u32(pg_w[s], reinterpret_cast<const uint8_t*>(base)),
                                           static_cast<int32_t>(i));
                                    ptr_s[s] += stride;
                                }
                            }
                        }

                        for (size_t k = 0; k < top_k; ++k) {
                            for (size_t s = 0; s < 4; ++s) {
                                const size_t lane_s = lane_base + s * vlen_w;
                                svst1(pg_w[s], dst_idx + k * args->sort_stride + lane_s, idx[k][s]);
                            }

                            if constexpr (std::is_same_v<T, int8_t>) {
                                const svint16_t pa = svqxtnt_s32(svqxtnb_s32(vals_s32[k][0]), vals_s32[k][1]);
                                const svint16_t pb = svqxtnt_s32(svqxtnb_s32(vals_s32[k][2]), vals_s32[k][3]);
                                const svint8_t res = svqxtnt_s16(svqxtnb_s16(pa), pb);
                                svst1(pg_b8, reinterpret_cast<int8_t*>(dst + k * args->sort_stride + lane_base), res);
                            } else {
                                const svuint16_t pa = svqxtnt_u32(svqxtnb_u32(vals_u32[k][0]), vals_u32[k][1]);
                                const svuint16_t pb = svqxtnt_u32(svqxtnb_u32(vals_u32[k][2]), vals_u32[k][3]);
                                const svuint8_t res = svqxtnt_u16(svqxtnb_u16(pa), pb);
                                svst1(pg_b8, reinterpret_cast<uint8_t*>(dst + k * args->sort_stride + lane_base), res);
                            }
                        }
                    };

                    size_t lane = 0;
                    for (; lane + vlen_b <= work_amount; lane += vlen_b) {
                        process_vec(lane, svptrue_b8());
                    }
                    if (lane < work_amount) {
                        process_vec(lane, svwhilelt_b8(lane, work_amount));
                    }
                    return;
                }
            }
#endif  // HAVE_SVE && HAVE_SVE2
            constexpr size_t V = 16;
            constexpr size_t kVecMax = 8;
            if (top_k <= kVecMax && work_amount >= V) {
                size_t lane = 0;
                for (; lane + V <= work_amount; lane += V) {
                    uint8x16_t vals[kVecMax];
                    int32x4_t idx0[kVecMax];
                    int32x4_t idx1[kVecMax];
                    int32x4_t idx2[kVecMax];
                    int32x4_t idx3[kVecMax];
                    for (size_t k = 0; k < top_k; ++k) {
                        const auto* base = src + k * args->sort_stride + lane;
                        if constexpr (std::is_same_v<T, int8_t>) {
                            vals[k] = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(base)));
                        } else {
                            vals[k] = vld1q_u8(reinterpret_cast<const uint8_t*>(base));
                        }
                        idx0[k] = vdupq_n_s32(static_cast<int32_t>(k));
                        idx1[k] = vdupq_n_s32(static_cast<int32_t>(k));
                        idx2[k] = vdupq_n_s32(static_cast<int32_t>(k));
                        idx3[k] = vdupq_n_s32(static_cast<int32_t>(k));
                    }

                    for (size_t i = 0; i + 1 < top_k; ++i) {
                        for (size_t j = top_k - 1; j > i; --j) {
                            uint8x16_t mask8;
                            if constexpr (std::is_same_v<T, int8_t>) {
                                const int8x16_t v_s8 = vreinterpretq_s8_u8(vals[j]);
                                const int8x16_t b_s8 = vreinterpretq_s8_u8(vals[j - 1]);
                                mask8 = mode_max ? vcgtq_s8(v_s8, b_s8) : vcltq_s8(v_s8, b_s8);
                            } else {
                                mask8 = mode_max ? vcgtq_u8(vals[j], vals[j - 1]) : vcltq_u8(vals[j], vals[j - 1]);
                            }
                            const uint8x16_t v_hi = vbslq_u8(mask8, vals[j], vals[j - 1]);
                            const uint8x16_t v_lo = vbslq_u8(mask8, vals[j - 1], vals[j]);
                            vals[j - 1] = v_hi;
                            vals[j] = v_lo;

                            uint32x4_t m0, m1, m2, m3;
                            expand_mask_u8(mask8, m0, m1, m2, m3);
                            const int32x4_t i0_hi = vbslq_s32(m0, idx0[j], idx0[j - 1]);
                            const int32x4_t i0_lo = vbslq_s32(m0, idx0[j - 1], idx0[j]);
                            const int32x4_t i1_hi = vbslq_s32(m1, idx1[j], idx1[j - 1]);
                            const int32x4_t i1_lo = vbslq_s32(m1, idx1[j - 1], idx1[j]);
                            const int32x4_t i2_hi = vbslq_s32(m2, idx2[j], idx2[j - 1]);
                            const int32x4_t i2_lo = vbslq_s32(m2, idx2[j - 1], idx2[j]);
                            const int32x4_t i3_hi = vbslq_s32(m3, idx3[j], idx3[j - 1]);
                            const int32x4_t i3_lo = vbslq_s32(m3, idx3[j - 1], idx3[j]);
                            idx0[j - 1] = i0_hi;
                            idx0[j] = i0_lo;
                            idx1[j - 1] = i1_hi;
                            idx1[j] = i1_lo;
                            idx2[j - 1] = i2_hi;
                            idx2[j] = i2_lo;
                            idx3[j - 1] = i3_hi;
                            idx3[j] = i3_lo;
                        }
                    }

                    size_t i = top_k;
                    const size_t stride = args->sort_stride;
                    const T* ptr = src + lane + top_k * stride;
                    auto insert_vec = [&](uint8x16_t v, int32_t idx) {
                        int32x4_t v_idx0 = vdupq_n_s32(idx);
                        int32x4_t v_idx1 = vdupq_n_s32(idx);
                        int32x4_t v_idx2 = vdupq_n_s32(idx);
                        int32x4_t v_idx3 = vdupq_n_s32(idx);

                        for (size_t k = 0; k < top_k; ++k) {
                            uint8x16_t mask8;
                            if constexpr (std::is_same_v<T, int8_t>) {
                                const int8x16_t v_s8 = vreinterpretq_s8_u8(v);
                                const int8x16_t b_s8 = vreinterpretq_s8_u8(vals[k]);
                                mask8 = mode_max ? vcgtq_s8(v_s8, b_s8) : vcltq_s8(v_s8, b_s8);
                            } else {
                                mask8 = mode_max ? vcgtq_u8(v, vals[k]) : vcltq_u8(v, vals[k]);
                            }

                            const uint8x16_t v_new = vbslq_u8(mask8, vals[k], v);
                            const uint8x16_t slot_new = vbslq_u8(mask8, v, vals[k]);
                            v = v_new;
                            vals[k] = slot_new;

                            uint32x4_t m0, m1, m2, m3;
                            expand_mask_u8(mask8, m0, m1, m2, m3);
                            const int32x4_t i_new0 = vbslq_s32(m0, idx0[k], v_idx0);
                            const int32x4_t i_new1 = vbslq_s32(m1, idx1[k], v_idx1);
                            const int32x4_t i_new2 = vbslq_s32(m2, idx2[k], v_idx2);
                            const int32x4_t i_new3 = vbslq_s32(m3, idx3[k], v_idx3);
                            const int32x4_t slot_i0 = vbslq_s32(m0, v_idx0, idx0[k]);
                            const int32x4_t slot_i1 = vbslq_s32(m1, v_idx1, idx1[k]);
                            const int32x4_t slot_i2 = vbslq_s32(m2, v_idx2, idx2[k]);
                            const int32x4_t slot_i3 = vbslq_s32(m3, v_idx3, idx3[k]);
                            v_idx0 = i_new0;
                            v_idx1 = i_new1;
                            v_idx2 = i_new2;
                            v_idx3 = i_new3;
                            idx0[k] = slot_i0;
                            idx1[k] = slot_i1;
                            idx2[k] = slot_i2;
                            idx3[k] = slot_i3;
                        }
                    };

                    for (; i + 1 < axis_dim; i += 2) {
                        const auto* base0 = ptr;
                        const auto* base1 = ptr + stride;
                        uint8x16_t v0;
                        uint8x16_t v1;
                        if constexpr (std::is_same_v<T, int8_t>) {
                            v0 = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(base0)));
                            v1 = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(base1)));
                        } else {
                            v0 = vld1q_u8(reinterpret_cast<const uint8_t*>(base0));
                            v1 = vld1q_u8(reinterpret_cast<const uint8_t*>(base1));
                        }
                        insert_vec(v0, static_cast<int32_t>(i));
                        insert_vec(v1, static_cast<int32_t>(i + 1));
                        ptr += 2 * stride;
                    }
                    for (; i < axis_dim; ++i) {
                        const auto* base = ptr;
                        uint8x16_t v;
                        if constexpr (std::is_same_v<T, int8_t>) {
                            v = vreinterpretq_u8_s8(vld1q_s8(reinterpret_cast<const int8_t*>(base)));
                        } else {
                            v = vld1q_u8(reinterpret_cast<const uint8_t*>(base));
                        }
                        insert_vec(v, static_cast<int32_t>(i));
                        ptr += stride;
                    }

                    for (size_t k = 0; k < top_k; ++k) {
                        auto* out = dst + k * args->sort_stride + lane;
                        auto* out_idx = dst_idx + k * args->sort_stride + lane;
                        if constexpr (std::is_same_v<T, int8_t>) {
                            vst1q_s8(reinterpret_cast<int8_t*>(out), vreinterpretq_s8_u8(vals[k]));
                        } else {
                            vst1q_u8(reinterpret_cast<uint8_t*>(out), vals[k]);
                        }
                        vst1q_s32(out_idx + 0, idx0[k]);
                        vst1q_s32(out_idx + 4, idx1[k]);
                        vst1q_s32(out_idx + 8, idx2[k]);
                        vst1q_s32(out_idx + 12, idx3[k]);
                    }
                }

                for (; lane < work_amount; ++lane) {
                    for (size_t i = 0; i < top_k; ++i) {
                        top_vals[i] = load_value(src, i, lane, jcp, args);
                        top_indices[i] = static_cast<int32_t>(i);
                    }

                    for (size_t i = 0; i + 1 < top_k; ++i) {
                        for (size_t j = top_k - 1; j > i; --j) {
                            const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                                      : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                            if (cmp) {
                                std::swap(top_vals[j], top_vals[j - 1]);
                                std::swap(top_indices[j], top_indices[j - 1]);
                            }
                        }
                    }

                    for (size_t i = top_k; i < axis_dim; ++i) {
                        top_vals[top_k] = load_value(src, i, lane, jcp, args);
                        top_indices[top_k] = static_cast<int32_t>(i);
                        for (size_t j = top_k; j > 0; --j) {
                            const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                                      : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                            if (cmp) {
                                std::swap(top_vals[j], top_vals[j - 1]);
                                std::swap(top_indices[j], top_indices[j - 1]);
                            } else {
                                break;
                            }
                        }
                    }

                    for (size_t i = 0; i < top_k; ++i) {
                        store_value(dst, i, lane, jcp, args, top_vals[i]);
                        store_index(dst_idx, i, lane, jcp, args, top_indices[i]);
                    }
                }
                return;
            }
        }
    }

    for (size_t lane = 0; lane < work_amount; ++lane) {
        for (size_t i = 0; i < top_k; ++i) {
            top_vals[i] = load_value(src, i, lane, jcp, args);
            top_indices[i] = static_cast<int32_t>(i);
        }

        for (size_t i = 0; i + 1 < top_k; ++i) {
            for (size_t j = top_k - 1; j > i; --j) {
                const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                          : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                if (cmp) {
                    std::swap(top_vals[j], top_vals[j - 1]);
                    std::swap(top_indices[j], top_indices[j - 1]);
                }
            }
        }

        for (size_t i = top_k; i < axis_dim; ++i) {
            top_vals[top_k] = load_value(src, i, lane, jcp, args);
            top_indices[top_k] = static_cast<int32_t>(i);
            for (size_t j = top_k; j > 0; --j) {
                const bool cmp = mode_max ? TopkCompare<T>::gt(top_vals[j], top_vals[j - 1])
                                          : TopkCompare<T>::lt(top_vals[j], top_vals[j - 1]);
                if (cmp) {
                    std::swap(top_vals[j], top_vals[j - 1]);
                    std::swap(top_indices[j], top_indices[j - 1]);
                } else {
                    break;
                }
            }
        }

        if (jcp->sort_index) {
            for (size_t i = 0; i + 1 < top_k; ++i) {
                for (size_t j = top_k - 1; j > i; --j) {
                    if (top_indices[j - 1] > top_indices[j]) {
                        std::swap(top_vals[j], top_vals[j - 1]);
                        std::swap(top_indices[j], top_indices[j - 1]);
                    }
                }
            }
        }

        for (size_t i = 0; i < top_k; ++i) {
            store_value(dst, i, lane, jcp, args, top_vals[i]);
            store_index(dst_idx, i, lane, jcp, args, top_indices[i]);
        }
    }
}

void topk_kernel_dispatch(const jit_topk_call_args* args) {
    const auto* jcp = args->config;
    if (!jcp) {
        return;
    }

    if (jcp->precision == ov::element::f32) {
        topk_kernel_impl<float>(args);
    } else if (jcp->precision == ov::element::f16) {
        topk_kernel_impl<ov::float16>(args);
    } else if (jcp->precision == ov::element::i32) {
        topk_kernel_impl<int32_t>(args);
    } else if (jcp->precision == ov::element::i8) {
        topk_kernel_impl<int8_t>(args);
    } else if (jcp->precision == ov::element::u8) {
        topk_kernel_impl<uint8_t>(args);
    } else {
        topk_kernel_impl<float>(args);
    }
}
}  // namespace

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
struct jit_uni_topk_kernel_aarch64 : public jit_uni_topk_kernel, public dnnl::impl::cpu::aarch64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_topk_kernel_aarch64)

    explicit jit_uni_topk_kernel_aarch64(jit_topk_config_params jcp)
        : jit_uni_topk_kernel(jcp),
          jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = ov::intel_cpu::jit_kernel_cast<decltype(ker_)>(jit_ker());
    }

    void generate() override {
        preamble();
        mov_imm(X_TMP_0, reinterpret_cast<size_t>(&topk_kernel_dispatch));
        blr(X_TMP_0);
        postamble();
    }
};

std::shared_ptr<jit_uni_topk_kernel> create_topk_kernel_aarch64(const jit_topk_config_params& jcp) {
    if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_512)) {
        return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::sve_512>>(jcp);
    }
    if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_384)) {
        return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::sve_384>>(jcp);
    }
    if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_256)) {
        return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::sve_256>>(jcp);
    }
    if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128)) {
        return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::sve_128>>(jcp);
    }
    if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd)) {
        return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::asimd>>(jcp);
    }
    return nullptr;
}

}  // namespace ov::intel_cpu::node

#endif  // defined(OPENVINO_ARCH_ARM64)
