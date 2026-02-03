// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "memory_format_filter.hpp"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/aarch64/brgemm_int8_kernel.hpp"
#include "nodes/kernels/aarch64/jit_int8_conv_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::aarch64 {

class BrgemmInt8ConvExecutor : public ov::intel_cpu::Executor {
public:
    BrgemmInt8ConvExecutor(const ConvAttrs& attrs,
                           const MemoryArgs& memory,
                           const ExecutorContext::CPtr& context);

    static bool supports(const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::brgconv_uni;
    }

private:
    bool execute_brgemm_1x1(PlainTensor& dst, const MemoryArgs& memory);
    void execute_impl(PlainTensor& dst, const MemoryArgs& memory);

    ConvAttrs attrs_;
    ov::element::Type bias_prec_ = ov::element::dynamic;
    std::vector<float> bias_f32_;
    std::shared_ptr<jit_int8_dot_kernel> kernel_u8_;
    std::shared_ptr<jit_int8_dot_kernel> kernel_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_1x4> kernel_block4_u8_;
    std::shared_ptr<jit_int8_brgemm_kernel_1x4> kernel_block4_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_1x4_dot> kernel_block4_dot_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_1x4_udot> kernel_block4_udot_;
    std::shared_ptr<jit_int8_brgemm_kernel_1x8_dot> kernel_block8_dot_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_1x8_udot> kernel_block8_udot_;
    std::shared_ptr<jit_int8_brgemm_kernel_1x8_dot_packed> kernel_block8_dot_packed_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_1x8_udot_packed> kernel_block8_udot_packed_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x4_dot> kernel_block4x4_dot_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x4_smmla_packed> kernel_block4x4_mmla_packed_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x8_smmla_packed> kernel_block4x8_mmla_packed_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x16_smmla_packed> kernel_block4x16_mmla_packed_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x4_udot> kernel_block4x4_udot_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x4_dot_packed> kernel_block4x4_dot_packed_s8_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x4_usmmla_packed> kernel_block4x4_mmla_packed_u8_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x8_usmmla_packed> kernel_block4x8_mmla_packed_u8_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x16_usmmla_packed> kernel_block4x16_mmla_packed_u8_;
    std::shared_ptr<jit_int8_brgemm_kernel_4x4_udot_packed> kernel_block4x4_udot_packed_;
    bool has_dotprod_ = false;
    bool has_i8mm_ = false;
    std::shared_ptr<BrgemmInt8Kernel> brgemm_1x1_u8_;
    std::shared_ptr<BrgemmInt8Kernel> brgemm_1x1_s8_;
    size_t brgemm_1x1_oc_ = 0;
    size_t brgemm_1x1_ic_ = 0;
    size_t brgemm_1x1_lda_ = 0;
    size_t brgemm_1x1_ldc_ = 0;
    static constexpr size_t brgemm_1x1_m_blk_ = 4;
    std::vector<int8_t> packed_wei_1x1_;
    std::vector<int8_t> packed_wei_1x1_col_;
    std::vector<int8_t> packed_wei_1x1_dot4_;
    std::vector<int8_t> packed_wei_1x1_dot8_;
    std::vector<int8_t> packed_wei_1x1_mmla4_;
    std::vector<int8_t> packed_wei_1x1_mmla8_;
    std::vector<int8_t> packed_wei_1x1_mmla16_;
    size_t packed_wei_1x1_dot4_stride_ = 0;
    size_t packed_wei_1x1_dot8_stride_ = 0;
    size_t packed_wei_1x1_mmla4_stride_ = 0;
    size_t packed_wei_1x1_mmla8_stride_ = 0;
    size_t packed_wei_1x1_mmla16_stride_ = 0;
    std::vector<int32_t> wei_comp_1x1_;
    const void* packed_wei_1x1_src_ = nullptr;
    size_t packed_wei_1x1_oc_ = 0;
    size_t packed_wei_1x1_ic_ = 0;
    std::vector<int8_t> packed_wei_brgemm_;
    std::vector<int8_t> packed_wei_brgemm_col_;
    std::vector<int8_t> packed_wei_brgemm_dot4_;
    std::vector<int8_t> packed_wei_brgemm_dot8_;
    std::vector<int8_t> packed_wei_brgemm_mmla4_;
    size_t packed_wei_brgemm_dot4_stride_ = 0;
    size_t packed_wei_brgemm_dot8_stride_ = 0;
    size_t packed_wei_brgemm_mmla4_stride_ = 0;
    std::vector<int8_t> packed_wei_brgemm_mmla8_;
    size_t packed_wei_brgemm_mmla8_stride_ = 0;
    std::vector<int8_t> packed_wei_brgemm_mmla16_;
    size_t packed_wei_brgemm_mmla16_stride_ = 0;
    std::vector<int8_t> packed_wei_brgemm_mmla4_fused_;
    size_t packed_wei_brgemm_mmla4_fused_stride_ = 0;
    std::vector<int8_t> packed_wei_brgemm_mmla8_fused_;
    size_t packed_wei_brgemm_mmla8_fused_stride_ = 0;
    std::vector<int8_t> packed_wei_brgemm_mmla16_fused_;
    size_t packed_wei_brgemm_mmla16_fused_stride_ = 0;
    size_t packed_wei_brgemm_mmla_fused_k_ = 0;
    std::vector<int32_t> wei_comp_brgemm_;
    const void* packed_wei_brgemm_src_ = nullptr;
    size_t packed_wei_brgemm_oc_ = 0;
    size_t packed_wei_brgemm_ic_ = 0;
    size_t packed_wei_brgemm_kh_ = 0;
    size_t packed_wei_brgemm_kw_ = 0;
    std::vector<int8_t> packed_wei_;
    const void* packed_wei_src_ = nullptr;
    size_t packed_oc_ = 0;
    size_t packed_ic_ = 0;
    size_t packed_kh_ = 0;
    size_t packed_kw_ = 0;
    PlainTensor tmp_dst_;
};

}  // namespace ov::intel_cpu::aarch64
