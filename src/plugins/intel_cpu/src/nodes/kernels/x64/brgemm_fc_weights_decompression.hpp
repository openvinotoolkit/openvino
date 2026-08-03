// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "jit_brgemm_decompress_kernel.hpp"
#include "jit_brgemm_src_quantization_kernel.hpp"
#include "jit_brgemm_weights_decompression_kernel.hpp"
#include "jit_fused_decomp_matmul_kernel.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

enum class WeightsDecompAlgo : uint8_t {
    IMMEDIATE,
    PREPACK,
};

struct BrgemmFCWeightsDecompressionConfig {
    size_t M;
    size_t N;
    size_t K;

    ov::element::Type src_dt;
    ov::element::Type wei_dt;
    ov::element::Type dst_dt;

    bool with_scales = false;
    bool broadcast_scales = false;
    size_t scales_ic_group_size = 0;

    bool with_zero_points = false;
    bool broadcast_zero_points = false;
    size_t zero_points_ic_group_size = 0;
    ov::element::Type zero_points_dt;

    bool with_src_dynamic_quant = false;
    size_t src_quant_group_size = 0;
    bool with_src_grouped_sum = false;
    size_t src_sum_group_size = 0;

    ov::element::Type scales_dt;

    WeightsDecompAlgo algo = WeightsDecompAlgo::PREPACK;
};

class BrgemmFCWeightsDecompression {
public:
    explicit BrgemmFCWeightsDecompression(const BrgemmFCWeightsDecompressionConfig& config);
    ~BrgemmFCWeightsDecompression();

    void execute(const void* src,
                 const void* weights,
                 void* dst,
                 const void* scales,
                 const void* zero_points,
                 void* scratchpad,
                 int num_threads) const;

    [[nodiscard]] size_t getScratchpadSize(int num_threads) const;

private:
    void executeFused(const void* src,
                      const void* weights,
                      void* dst,
                      const void* scales,
                      const void* zero_points,
                      void* scratchpad,
                      int num_threads) const;

    void executePrepack(const void* src,
                        const void* weights,
                        void* dst,
                        const void* scales,
                        const void* zero_points,
                        void* scratchpad,
                        int num_threads) const;

    bool initFusedKernels(const BrgemmFCWeightsDecompressionConfig& config,
                          dnnl::impl::data_type_t dnnl_wei_dt,
                          dnnl::impl::data_type_t dnnl_src_dt);

    void initPrepackKernels(const BrgemmFCWeightsDecompressionConfig& config,
                            dnnl::impl::data_type_t dnnl_wei_dt,
                            dnnl::impl::data_type_t dnnl_src_dt);

    void performSrcQuantization(const void* src,
                                int8_t* qsrc,
                                float* src_dscales,
                                int32_t* src_grouped_sum,
                                const BrgemmFCWeightsDecompressionConfig& cfg,
                                size_t ic_groups) const;

    BrgemmFCWeightsDecompressionConfig m_config;
    std::unique_ptr<jit_weights_decompression_kernel_base_t> m_wei_decomp_kernel;
    std::unique_ptr<jit_src_quantization_kernel_base_t> m_src_quant_kernel;

    // Fused decompression+matmul kernel (immediate algo - standalone JIT)
    std::unique_ptr<jit_fused_decomp_matmul_kernel_base_t> m_fused_kernel;

    // Standard brgemm kernels (prepack algo - decompress then matmul)
    brgemm_kernel_t* m_brg_kernel = nullptr;
    brgemm_kernel_t* m_brg_kernel_ic_tail = nullptr;
    brgemm_desc_t m_brg_desc;
    brgemm_desc_t m_brg_desc_ic_tail;

    size_t m_oc_block = 0;
    size_t m_ic_block = 0;
    size_t m_nb_ic_blocking = 1;
    size_t m_simd_w = 0;
    bool m_use_fused_kernel = false;
};

}  // namespace ov::intel_cpu
