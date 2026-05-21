// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// NOLINTBEGIN(*)

#include "brgemm_fc_weights_decompression.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/c_types_map.hpp>
#include <common/dnnl_thread.hpp>
#include <common/type_helpers.hpp>
#include <common/utils.hpp>
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;

static data_type_t to_dnnl_dt(ov::element::Type et) {
    if (et == ov::element::f32)
        return data_type::f32;
    if (et == ov::element::bf16)
        return data_type::bf16;
    if (et == ov::element::f16)
        return data_type::f16;
    if (et == ov::element::i8)
        return data_type::s8;
    if (et == ov::element::u8)
        return data_type::u8;
    if (et == ov::element::u4)
        return data_type::u4;
    if (et == ov::element::i4)
        return data_type::s4;
    if (et == ov::element::nf4)
        return data_type::nf4;
    if (et == ov::element::f4e2m1)
        return data_type::f4_e2m1;
    if (et == ov::element::f8e8m0)
        return data_type::e8m0;
    if (et == ov::element::u2)
        return data_type::u2;
    return data_type::undef;
}

static size_t get_typesize_scale(data_type_t dt) {
    if (dt == data_type::u2)
        return 4;
    if (one_of(dt, data_type::nf4, data_type::s4, data_type::u4, data_type::f4_e2m1))
        return 2;
    return 1;
}

BrgemmFCWeightsDecompression::~BrgemmFCWeightsDecompression() {
    if (m_brg_kernel)
        brgemm_kernel_destroy(m_brg_kernel);
    if (m_brg_kernel_ic_tail)
        brgemm_kernel_destroy(m_brg_kernel_ic_tail);
}

BrgemmFCWeightsDecompression::BrgemmFCWeightsDecompression(const BrgemmFCWeightsDecompressionConfig& config)
    : m_config(config) {
    m_simd_w = mayiuse(avx512_core) ? 16 : 8;
    m_oc_block = m_simd_w <= 8 ? std::min<size_t>(32, config.N) : std::min<size_t>(64, config.N);
    m_ic_block = m_simd_w <= 8 ? 32 : 64;

    auto dnnl_wei_dt = to_dnnl_dt(config.wei_dt);
    auto dnnl_src_dt = to_dnnl_dt(config.src_dt);

    if (config.algo == WeightsDecompAlgo::IMMEDIATE) {
        // Try to create fused brgemm kernel that does decompression inline
        m_use_fused_kernel = initFusedKernels(config, dnnl_wei_dt, dnnl_src_dt);
    }

    if (!m_use_fused_kernel) {
        // Fallback to prepack (decompress-then-matmul) path
        initPrepackKernels(config, dnnl_wei_dt, dnnl_src_dt);
    }

    if (config.with_src_dynamic_quant) {
        src_quantization_compile_params_t jcp = {};
        jcp.ic_quant_block = config.src_quant_group_size;
        jcp.with_src_grouped_sum = config.with_src_grouped_sum;
        jcp.src_sum_group_size =
            config.src_sum_group_size > 0 ? config.src_sum_group_size : config.src_quant_group_size;
        jcp.src_dt = dnnl_src_dt;
        jcp.qsrc_dt = data_type::s8;

        if (mayiuse(avx512_core)) {
            m_src_quant_kernel = std::make_unique<jit_brgemm_src_quantization_kernel_t<avx512_core>>(jcp);
        } else if (mayiuse(avx2)) {
            m_src_quant_kernel = std::make_unique<jit_brgemm_src_quantization_kernel_t<avx2>>(jcp);
        }
    }
}

bool BrgemmFCWeightsDecompression::initFusedKernels(const BrgemmFCWeightsDecompressionConfig& config,
                                                     data_type_t dnnl_wei_dt,
                                                     data_type_t dnnl_src_dt) {
    fused_decomp_matmul_compile_params_t jcp = {};
    jcp.oc_block = m_oc_block;
    jcp.src_dt = config.with_src_dynamic_quant ? data_type::s8 : dnnl_src_dt;
    jcp.wei_dt = dnnl_wei_dt;
    jcp.dst_dt = data_type::f32;

    jcp.with_scales = config.with_scales;
    jcp.broadcast_scales = config.broadcast_scales;
    jcp.scales_dt = (config.scales_dt != ov::element::Type()) ? to_dnnl_dt(config.scales_dt) : data_type::f32;

    jcp.with_zero_points = config.with_zero_points;
    jcp.broadcast_zero_points = config.broadcast_zero_points;
    jcp.zero_points_dt = to_dnnl_dt(config.zero_points_dt);
    if (jcp.zero_points_dt == data_type::undef)
        jcp.zero_points_dt = data_type::u8;

    jcp.is_dyn_quant = config.with_src_dynamic_quant;
    jcp.with_src_grouped_sum = config.with_src_grouped_sum;

    // Compute ic_block: base block depends on weight type, then align to group sizes
    size_t base_block;
    if (dnnl_wei_dt == data_type::u2) {
        base_block = 64;
    } else if (one_of(dnnl_wei_dt, data_type::nf4, data_type::s4, data_type::u4, data_type::f4_e2m1)) {
        base_block = 32;
    } else {
        base_block = 16;
    }

    if (config.scales_ic_group_size > 0)
        base_block = std::min(base_block, config.scales_ic_group_size);
    if (config.zero_points_ic_group_size > 0)
        base_block = std::min(base_block, config.zero_points_ic_group_size);
    if (config.with_src_dynamic_quant && config.src_quant_group_size > 0)
        base_block = std::min(base_block, config.src_quant_group_size);

    jcp.ic_block = base_block;
    m_ic_block = base_block;
    m_nb_ic_blocking = (config.K + m_ic_block - 1) / m_ic_block;

    try {
        if (mayiuse(avx512_core)) {
            auto kernel = std::make_unique<jit_fused_decomp_matmul_kernel_t<avx512_core>>(jcp);
            m_fused_kernel = std::move(kernel);
        } else if (mayiuse(avx2)) {
            auto kernel = std::make_unique<jit_fused_decomp_matmul_kernel_t<avx2>>(jcp);
            m_fused_kernel = std::move(kernel);
        }
    } catch (...) {
        m_fused_kernel = nullptr;
    }

    return m_fused_kernel != nullptr;
}

void BrgemmFCWeightsDecompression::initPrepackKernels(const BrgemmFCWeightsDecompressionConfig& config,
                                                       data_type_t dnnl_wei_dt,
                                                       data_type_t dnnl_src_dt) {
    weights_decompression_compile_params_t jcp = {};
    jcp.oc_size = m_oc_block;

    if (dnnl_wei_dt == data_type::u2) {
        jcp.ic_internal_size = 4;
    } else if (dnnl_src_dt == data_type::bf16 ||
               one_of(dnnl_wei_dt, data_type::nf4, data_type::s4, data_type::u4, data_type::f4_e2m1)) {
        jcp.ic_internal_size = 2;
    } else {
        jcp.ic_internal_size = 1;
    }
    jcp.with_scales = config.with_scales;
    jcp.broadcast_scales = config.broadcast_scales;
    jcp.with_zero_points = config.with_zero_points;
    jcp.broadcast_zero_points = config.broadcast_zero_points;
    jcp.weights_dt = dnnl_wei_dt;
    jcp.decomp_buffer_dt = dnnl_src_dt;
    jcp.scales_dt = (config.scales_dt != ov::element::Type()) ? to_dnnl_dt(config.scales_dt) : data_type::f32;
    jcp.zero_points_dt = to_dnnl_dt(config.zero_points_dt);

    if (mayiuse(avx512_core)) {
        m_wei_decomp_kernel = std::make_unique<jit_brgemm_weights_decompression_kernel_t<avx512_core>>(jcp);
    } else if (mayiuse(avx2)) {
        m_wei_decomp_kernel = std::make_unique<jit_brgemm_weights_decompression_kernel_t<avx2>>(jcp);
    }

    // Initialize brgemm kernels for the matmul after decompression
    auto brg_dt_a = config.with_src_dynamic_quant ? data_type::s8 : dnnl_src_dt;
    auto brg_dt_b = dnnl_src_dt;  // decompressed weights are in src_dt

    size_t ic_block = m_ic_block;
    size_t ic_tail = config.K % ic_block;

    brgemm_desc_init(&m_brg_desc,
                     cpu_isa_t::isa_undef,
                     brgemm_addr,
                     brg_dt_a,
                     brg_dt_b,
                     false, false,
                     brgemm_row_major,
                     1.0F, 1.0F,
                     static_cast<dim_t>(config.K),
                     static_cast<dim_t>(m_oc_block),
                     static_cast<dim_t>(config.N),
                     1,
                     static_cast<dim_t>(m_oc_block),
                     static_cast<dim_t>(ic_block));
    brgemm_kernel_create(&m_brg_kernel, m_brg_desc);

    if (ic_tail > 0) {
        brgemm_desc_init(&m_brg_desc_ic_tail,
                         cpu_isa_t::isa_undef,
                         brgemm_addr,
                         brg_dt_a,
                         brg_dt_b,
                         false, false,
                         brgemm_row_major,
                         1.0F, 1.0F,
                         static_cast<dim_t>(config.K),
                         static_cast<dim_t>(m_oc_block),
                         static_cast<dim_t>(config.N),
                         1,
                         static_cast<dim_t>(m_oc_block),
                         static_cast<dim_t>(ic_tail));
        brgemm_kernel_create(&m_brg_kernel_ic_tail, m_brg_desc_ic_tail);
    }
}

size_t BrgemmFCWeightsDecompression::getScratchpadSize(int num_threads) const {
    size_t decomp_buf_size = 0;
    if (!m_use_fused_kernel && m_wei_decomp_kernel) {
        decomp_buf_size =
            static_cast<size_t>(num_threads) * m_ic_block * m_nb_ic_blocking * m_oc_block * m_config.src_dt.size();
    }

    size_t qsrc_size = 0;
    size_t src_scales_size = 0;
    size_t src_grouped_sum_size = 0;
    size_t tmp_acc_size = 0;
    if (m_config.with_src_dynamic_quant) {
        qsrc_size = m_config.M * m_config.K * sizeof(int8_t);
        size_t ic_groups = (m_config.K + m_config.src_quant_group_size - 1) / m_config.src_quant_group_size;
        src_scales_size = m_config.M * ic_groups * sizeof(float);
        if (m_config.with_src_grouped_sum && m_config.src_sum_group_size > 0) {
            size_t ic_sum_groups = (m_config.K + m_config.src_sum_group_size - 1) / m_config.src_sum_group_size;
            src_grouped_sum_size = m_config.M * ic_sum_groups * sizeof(int32_t);
        }
        if (!m_use_fused_kernel) {
            tmp_acc_size = static_cast<size_t>(num_threads) * m_oc_block * sizeof(float);
        }
    }

    return decomp_buf_size + qsrc_size + src_scales_size + src_grouped_sum_size + tmp_acc_size;
}

void BrgemmFCWeightsDecompression::execute(const void* src,
                                           const void* weights,
                                           void* dst,
                                           const void* scales,
                                           const void* zero_points,
                                           void* scratchpad,
                                           int num_threads) const {
    if (m_use_fused_kernel) {
        executeFused(src, weights, dst, scales, zero_points, scratchpad, num_threads);
    } else {
        executePrepack(src, weights, dst, scales, zero_points, scratchpad, num_threads);
    }
}

void BrgemmFCWeightsDecompression::executeFused(const void* src,
                                                const void* weights,
                                                void* dst,
                                                const void* scales,
                                                const void* zero_points,
                                                void* scratchpad,
                                                int num_threads) const {
    const auto& cfg = m_config;
    auto dnnl_wei_dt = to_dnnl_dt(cfg.wei_dt);
    const size_t typesize_scale = get_typesize_scale(dnnl_wei_dt);

    char* scratch_buf = static_cast<char*>(scratchpad);
    size_t offset = 0;

    int8_t* qsrc = nullptr;
    float* src_dscales = nullptr;
    int32_t* src_grouped_sum = nullptr;
    size_t ic_groups = 0;
    size_t ic_sum_groups = 0;

    if (cfg.with_src_dynamic_quant) {
        qsrc = reinterpret_cast<int8_t*>(scratch_buf + offset);
        offset += cfg.M * cfg.K * sizeof(int8_t);

        ic_groups = (cfg.K + cfg.src_quant_group_size - 1) / cfg.src_quant_group_size;
        src_dscales = reinterpret_cast<float*>(scratch_buf + offset);
        offset += cfg.M * ic_groups * sizeof(float);

        if (cfg.with_src_grouped_sum && cfg.src_sum_group_size > 0) {
            ic_sum_groups = (cfg.K + cfg.src_sum_group_size - 1) / cfg.src_sum_group_size;
            src_grouped_sum = reinterpret_cast<int32_t*>(scratch_buf + offset);
            offset += cfg.M * ic_sum_groups * sizeof(int32_t);
        }

        performSrcQuantization(src, qsrc, src_dscales, src_grouped_sum, cfg, ic_groups);
    }

    const size_t nb_oc = (cfg.N + m_oc_block - 1) / m_oc_block;
    const size_t nb_ic = (cfg.K + m_ic_block - 1) / m_ic_block;

    const auto* weights_u8 = static_cast<const uint8_t*>(weights);
    const auto* scales_u8 = static_cast<const uint8_t*>(scales);
    const auto* zp_u8 = static_cast<const uint8_t*>(zero_points);
    auto* dst_f32 = static_cast<float*>(dst);

    const size_t scales_dt_size =
        (cfg.scales_dt != ov::element::Type() && cfg.scales_dt.size() > 0) ? cfg.scales_dt.size() : sizeof(float);
    const size_t zp_dt_size = (cfg.zero_points_dt != ov::element::Type() && cfg.zero_points_dt.size() > 0)
                                  ? cfg.zero_points_dt.size()
                                  : sizeof(float);

    // Weight layout is [OC, IC] with sub-byte packing.
    // Scales/ZP layout: per-group is [IC_groups, OC] or broadcast.
    parallel_nd(static_cast<dim_t>(cfg.M), static_cast<dim_t>(nb_oc), [&](dim_t mb, dim_t ocb_idx) {
        size_t oc = static_cast<size_t>(ocb_idx) * m_oc_block;
        float* dst_ptr = dst_f32 + mb * cfg.N + oc;

        for (size_t icb_idx = 0; icb_idx < nb_ic; icb_idx++) {
            size_t ic = icb_idx * m_ic_block;
            size_t cur_ic_size = std::min(m_ic_block, cfg.K - ic);

            // Source pointer for this IC block
            const void* src_ptr;
            if (cfg.with_src_dynamic_quant) {
                src_ptr = qsrc + mb * cfg.K + ic;
            } else {
                src_ptr = static_cast<const char*>(src) + (mb * cfg.K + ic) * cfg.src_dt.size();
            }

            // Weight pointer: [OC, IC] layout, offset to (oc, ic)
            const void* wei_ptr = weights_u8 + (oc * cfg.K + ic) / typesize_scale;

            // Scales pointer for this IC group
            const void* wei_scales_ptr = nullptr;
            if (cfg.with_scales) {
                if (cfg.broadcast_scales) {
                    wei_scales_ptr = scales_u8;
                } else if (cfg.scales_ic_group_size == 0 || cfg.scales_ic_group_size >= cfg.K) {
                    wei_scales_ptr = scales_u8 + oc * scales_dt_size;
                } else {
                    size_t scale_group = ic / cfg.scales_ic_group_size;
                    wei_scales_ptr = scales_u8 + (scale_group * cfg.N + oc) * scales_dt_size;
                }
            }

            // Zero points pointer for this IC group
            const void* wei_zp_ptr = nullptr;
            if (cfg.with_zero_points) {
                if (cfg.broadcast_zero_points) {
                    wei_zp_ptr = zp_u8;
                } else if (cfg.zero_points_ic_group_size == 0 || cfg.zero_points_ic_group_size >= cfg.K) {
                    wei_zp_ptr = zp_u8 + oc * zp_dt_size;
                } else {
                    size_t zp_group = ic / cfg.zero_points_ic_group_size;
                    wei_zp_ptr = zp_u8 + (zp_group * cfg.N + oc) * zp_dt_size;
                }
            }

            // Source scales and grouped sum for this IC group
            const void* src_scales_ptr = nullptr;
            const void* src_sum_ptr = nullptr;
            if (cfg.with_src_dynamic_quant && src_dscales) {
                size_t quant_group = ic / cfg.src_quant_group_size;
                src_scales_ptr = src_dscales + mb * ic_groups + quant_group;
                if (src_grouped_sum && cfg.src_sum_group_size > 0) {
                    size_t sum_group = ic / cfg.src_sum_group_size;
                    src_sum_ptr = src_grouped_sum + mb * ic_sum_groups + sum_group;
                }
            }

            fused_decomp_matmul_runtime_params_t rt_params = {};
            rt_params.src_ptr = src_ptr;
            rt_params.wei_ptr = wei_ptr;
            rt_params.dst_ptr = dst_ptr;
            rt_params.scales_ptr = wei_scales_ptr;
            rt_params.zero_points_ptr = wei_zp_ptr;
            rt_params.src_scales_ptr = src_scales_ptr;
            rt_params.src_grouped_sum_ptr = src_sum_ptr;
            rt_params.ic_size = cur_ic_size;
            rt_params.is_accumulate = (icb_idx > 0) ? 1 : 0;

            (*m_fused_kernel)(&rt_params);
        }
    });
}

void BrgemmFCWeightsDecompression::executePrepack(const void* src,
                                                   const void* weights,
                                                   void* dst,
                                                   const void* scales,
                                                   const void* zero_points,
                                                   void* scratchpad,
                                                   int num_threads) const {
    const auto& cfg = m_config;
    auto dnnl_wei_dt = to_dnnl_dt(cfg.wei_dt);
    const size_t typesize_scale = get_typesize_scale(dnnl_wei_dt);

    const size_t ic_internal_block = [&] {
        if (dnnl_wei_dt == data_type::u2)
            return size_t(4);
        if (to_dnnl_dt(cfg.src_dt) == data_type::bf16 ||
            one_of(dnnl_wei_dt, data_type::nf4, data_type::s4, data_type::u4, data_type::f4_e2m1))
            return size_t(2);
        return size_t(1);
    }();

    char* decomp_buf = static_cast<char*>(scratchpad);
    int8_t* qsrc = nullptr;
    float* src_dscales = nullptr;
    int32_t* src_grouped_sum = nullptr;
    float* tmp_acc = nullptr;

    size_t decomp_buf_total =
        static_cast<size_t>(num_threads) * m_ic_block * m_nb_ic_blocking * m_oc_block * cfg.src_dt.size();

    if (cfg.with_src_dynamic_quant && m_src_quant_kernel) {
        qsrc = reinterpret_cast<int8_t*>(decomp_buf + decomp_buf_total);
        size_t qsrc_total = cfg.M * cfg.K * sizeof(int8_t);
        src_dscales = reinterpret_cast<float*>(decomp_buf + decomp_buf_total + qsrc_total);
        size_t ic_groups = (cfg.K + cfg.src_quant_group_size - 1) / cfg.src_quant_group_size;
        size_t src_scales_size_total = cfg.M * ic_groups * sizeof(float);

        if (cfg.with_src_grouped_sum && cfg.src_sum_group_size > 0) {
            src_grouped_sum =
                reinterpret_cast<int32_t*>(decomp_buf + decomp_buf_total + qsrc_total + src_scales_size_total);
            size_t ic_sum_groups = (cfg.K + cfg.src_sum_group_size - 1) / cfg.src_sum_group_size;
            size_t src_grouped_sum_size_total = cfg.M * ic_sum_groups * sizeof(int32_t);
            tmp_acc = reinterpret_cast<float*>(decomp_buf + decomp_buf_total + qsrc_total + src_scales_size_total +
                                               src_grouped_sum_size_total);
        } else {
            tmp_acc = reinterpret_cast<float*>(decomp_buf + decomp_buf_total + qsrc_total + src_scales_size_total);
        }

        size_t ic_groups_val = ic_groups;
        performSrcQuantization(src, qsrc, src_dscales, src_grouped_sum, cfg, ic_groups_val);
    }

    if (m_wei_decomp_kernel) {
        size_t decomp_buf_per_thr = m_ic_block * m_nb_ic_blocking * m_oc_block * cfg.src_dt.size();
        size_t nb_oc = (cfg.N + m_oc_block - 1) / m_oc_block;
        size_t nb_ic = (cfg.K + m_ic_block - 1) / m_ic_block;

        const auto* weights_u8 = static_cast<const uint8_t*>(weights);
        const auto* scales_u8 = static_cast<const uint8_t*>(scales);
        const auto* zp_u8 = static_cast<const uint8_t*>(zero_points);
        const size_t scales_dt_size =
            (cfg.scales_dt != ov::element::Type() && cfg.scales_dt.size() > 0) ? cfg.scales_dt.size() : sizeof(float);
        size_t zp_dt_size = (cfg.zero_points_dt != ov::element::Type() && cfg.zero_points_dt.size() > 0)
                                ? cfg.zero_points_dt.size()
                                : sizeof(float);

        auto* dst_f32 = static_cast<float*>(dst);

        parallel_nd(static_cast<dim_t>(cfg.M), static_cast<dim_t>(nb_oc), [&](dim_t mb, dim_t ocb_idx) {
            const auto ithr = static_cast<size_t>(parallel_get_thread_num());
            auto* local_decomp_buf = decomp_buf + ithr * decomp_buf_per_thr;

            size_t oc = static_cast<size_t>(ocb_idx) * m_oc_block;
            size_t cur_oc_size = std::min(m_oc_block, cfg.N - oc);
            float* dst_ptr = dst_f32 + mb * cfg.N + oc;

            if (cfg.with_src_dynamic_quant && tmp_acc != nullptr) {
                auto* local_tmp_acc = tmp_acc + ithr * m_oc_block;
                std::fill(dst_ptr, dst_ptr + cur_oc_size, 0.0F);
                for (size_t icb_idx = 0; icb_idx < nb_ic; icb_idx++) {
                    size_t ic = icb_idx * m_ic_block;
                    size_t cur_ic_size = std::min(m_ic_block, cfg.K - ic);
                    size_t ic_internal_size = cur_ic_size / ic_internal_block;

                    size_t w_offset = (ic * cfg.N + oc * cur_ic_size) * cfg.wei_dt.size() / typesize_scale;
                    const auto* weights_ptr = weights_u8 + w_offset;

                    const uint8_t* wei_scales_ptr = scales_u8;
                    if (cfg.with_scales && !cfg.broadcast_scales) {
                        size_t scale_group = cfg.scales_ic_group_size > 0 ? ic / cfg.scales_ic_group_size : 0;
                        wei_scales_ptr = scales_u8 + (scale_group * cfg.N + oc) * scales_dt_size;
                    }

                    const uint8_t* wei_zp_ptr = zp_u8;
                    if (cfg.with_zero_points && !cfg.broadcast_zero_points) {
                        size_t zp_group = cfg.zero_points_ic_group_size > 0 ? ic / cfg.zero_points_ic_group_size : 0;
                        wei_zp_ptr = zp_u8 + (zp_group * cfg.N + oc) * zp_dt_size;
                    }

                    weights_decompression_runtime_params_t rt_params = {};
                    rt_params.weights_ptr = weights_ptr;
                    rt_params.decomp_buffer_ptr = local_decomp_buf;
                    rt_params.scales_ptr = wei_scales_ptr;
                    rt_params.zero_points_ptr = wei_zp_ptr;
                    rt_params.ic_size = ic_internal_size;
                    (*m_wei_decomp_kernel)(&rt_params);

                    const void* src_ptr = static_cast<const void*>(qsrc + mb * cfg.K + ic);
                    bool is_ic_tail = (cur_ic_size < m_ic_block);
                    auto* brg_kernel = is_ic_tail ? m_brg_kernel_ic_tail : m_brg_kernel;
                    if (!brg_kernel) continue;

                    std::fill(local_tmp_acc, local_tmp_acc + cur_oc_size, 0.0F);
                    brgemm_batch_element_t batch_elem;
                    batch_elem.ptr.A = src_ptr;
                    batch_elem.ptr.B = local_decomp_buf;
                    brgemm_kernel_execute(brg_kernel, 1, &batch_elem, static_cast<void*>(local_tmp_acc));

                    const size_t ic_group = ic / cfg.src_quant_group_size;
                    const size_t total_groups = (cfg.K + cfg.src_quant_group_size - 1) / cfg.src_quant_group_size;
                    const float dscale = src_dscales[mb * total_groups + ic_group];
                    for (size_t oc_i = 0; oc_i < cur_oc_size; oc_i++) {
                        dst_ptr[oc_i] += local_tmp_acc[oc_i] * dscale;
                    }
                }
                return;
            }

            for (size_t icb_idx = 0; icb_idx < nb_ic; icb_idx++) {
                size_t ic = icb_idx * m_ic_block;
                size_t cur_ic_size = std::min(m_ic_block, cfg.K - ic);
                size_t ic_internal_size = cur_ic_size / ic_internal_block;

                size_t w_offset = (ic * cfg.N + oc * cur_ic_size) * cfg.wei_dt.size() / typesize_scale;
                const auto* weights_ptr = weights_u8 + w_offset;

                const uint8_t* wei_scales_ptr = scales_u8;
                if (cfg.with_scales && !cfg.broadcast_scales) {
                    size_t scale_group = cfg.scales_ic_group_size > 0 ? ic / cfg.scales_ic_group_size : 0;
                    wei_scales_ptr = scales_u8 + (scale_group * cfg.N + oc) * scales_dt_size;
                }

                const uint8_t* wei_zp_ptr = zp_u8;
                if (cfg.with_zero_points && !cfg.broadcast_zero_points) {
                    size_t zp_group = cfg.zero_points_ic_group_size > 0 ? ic / cfg.zero_points_ic_group_size : 0;
                    wei_zp_ptr = zp_u8 + (zp_group * cfg.N + oc) * zp_dt_size;
                }

                weights_decompression_runtime_params_t rt_params = {};
                rt_params.weights_ptr = weights_ptr;
                rt_params.decomp_buffer_ptr = local_decomp_buf;
                rt_params.scales_ptr = wei_scales_ptr;
                rt_params.zero_points_ptr = wei_zp_ptr;
                rt_params.ic_size = ic_internal_size;
                (*m_wei_decomp_kernel)(&rt_params);

                const void* src_ptr = cfg.with_src_dynamic_quant
                                          ? static_cast<const void*>(qsrc + mb * cfg.K + ic)
                                          : static_cast<const void*>(static_cast<const char*>(src) +
                                                                     (mb * cfg.K + ic) * cfg.src_dt.size());
                bool is_ic_tail = (cur_ic_size < m_ic_block);
                auto* brg_kernel = is_ic_tail ? m_brg_kernel_ic_tail : m_brg_kernel;

                if (brg_kernel) {
                    if (icb_idx == 0) {
                        std::fill(dst_ptr, dst_ptr + cur_oc_size, 0.0F);
                    }
                    brgemm_batch_element_t batch_elem;
                    batch_elem.ptr.A = src_ptr;
                    batch_elem.ptr.B = local_decomp_buf;
                    brgemm_kernel_execute(brg_kernel, 1, &batch_elem, static_cast<void*>(dst_ptr));
                } else {
                    const auto* decomp_f32 = reinterpret_cast<const float*>(local_decomp_buf);
                    for (size_t oc_i = 0; oc_i < cur_oc_size; oc_i++) {
                        float acc = (icb_idx == 0) ? 0.0F : dst_ptr[oc_i];
                        for (size_t ic_i = 0; ic_i < cur_ic_size; ic_i++) {
                            float src_val = cfg.with_src_dynamic_quant
                                                ? static_cast<float>(static_cast<const int8_t*>(src_ptr)[ic_i])
                                                : static_cast<const float*>(src_ptr)[ic_i];
                            acc += src_val * decomp_f32[ic_i * cur_oc_size + oc_i];
                        }
                        dst_ptr[oc_i] = acc;
                    }
                }
            }
        });
    }
}

void BrgemmFCWeightsDecompression::performSrcQuantization(const void* src,
                                                           int8_t* qsrc,
                                                           float* src_dscales,
                                                           int32_t* src_grouped_sum,
                                                           const BrgemmFCWeightsDecompressionConfig& cfg,
                                                           size_t ic_groups) const {
    if (!m_src_quant_kernel) return;

    size_t vec_loop_end = (ic_groups - 1) * cfg.src_quant_group_size;
    const auto* src_f32 = static_cast<const float*>(src);

    parallel_nd(static_cast<dim_t>(cfg.M), [&](dim_t mb) {
        src_quantization_runtime_params_t rt_params = {};
        rt_params.src_ptr = src_f32 + mb * cfg.K;
        rt_params.qsrc_ptr = qsrc + mb * cfg.K;
        rt_params.src_scales_ptr = src_dscales + mb * ic_groups;
        if (cfg.with_src_grouped_sum && src_grouped_sum) {
            const size_t ic_sum_groups = (cfg.K + cfg.src_sum_group_size - 1) / cfg.src_sum_group_size;
            rt_params.src_grouped_sum_ptr = src_grouped_sum + mb * ic_sum_groups;
        } else {
            rt_params.src_grouped_sum_ptr = nullptr;
        }
        rt_params.ic_size = vec_loop_end;
        (*m_src_quant_kernel)(&rt_params);

        // Handle tail elements not covered by JIT kernel
        if (vec_loop_end != cfg.K) {
            float amax = 0;
            for (size_t ic = vec_loop_end; ic < cfg.K; ic++) {
                amax = std::max(amax, std::abs(src_f32[mb * cfg.K + ic]));
            }
            const float dscale = amax / 127.0F;
            const float qscale = (dscale != 0) ? (1.0F / dscale) : 0;
            src_dscales[mb * ic_groups + ic_groups - 1] = dscale;
            for (size_t ic = vec_loop_end; ic < cfg.K; ic++) {
                qsrc[mb * cfg.K + ic] = static_cast<int8_t>(std::round(src_f32[mb * cfg.K + ic] * qscale));
            }
        }

        // Compute grouped sums for zero-point compensation
        if (cfg.with_src_grouped_sum && src_grouped_sum && cfg.src_sum_group_size > 0) {
            const size_t ic_sum_groups = (cfg.K + cfg.src_sum_group_size - 1) / cfg.src_sum_group_size;
            auto* mb_sums = src_grouped_sum + mb * ic_sum_groups;
            for (size_t g = 0; g < ic_sum_groups; g++) {
                const size_t ic_begin = g * cfg.src_sum_group_size;
                const size_t ic_end = std::min(cfg.K, (g + 1) * cfg.src_sum_group_size);
                int32_t sum = 0;
                for (size_t ic = ic_begin; ic < ic_end; ic++) {
                    sum += qsrc[mb * cfg.K + ic];
                }
                mb_sums[g] = sum;
            }
        }
    });
}

}  // namespace ov::intel_cpu
// NOLINTEND(*)
