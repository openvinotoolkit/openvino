// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/rt_info.hpp>
#include <ie_ngraph_utils.hpp>

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/expression.hpp"

#include "jit_emitter.hpp"
#include "jit_load_store_emitters.hpp"

#include "transformations/snippets/x64/op/store_convert.hpp"
// Matmul support:
#include "libxsmm.h"

namespace ov {
namespace intel_cpu {
//todo: this class is largely a copy of BrgemmEmitter. It makes sense to develop a base class for
// BrgemmEmitter and BrgemmTppEmitter, but his can't be done until PR#19335 is merged, since it drastically
// simplifies BrgemmEmitter implementation.
class BrgemmTppEmitter : public jit_emitter {
public:
    BrgemmTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                  dnnl::impl::cpu::x64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return m_with_scratch ? 3 : 2; }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);
    size_t aux_gprs_count() const override;

    static size_t get_in_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);
    static size_t get_out_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    struct brgemmCtx {
        brgemmCtx() : M(0), N(0), K(0),
                    LDA(0), LDB(0), LDC(0),
                    dt_in0(dnnl_f32), dt_in1(dnnl_f32),
                    is_with_amx(false), is_with_comp(false), beta(0) {}
        size_t M, N, K, LDA, LDB, LDC;
        dnnl_data_type_t dt_in0, dt_in1;
        char palette[64] = {};
        bool is_with_amx;
        bool is_with_comp;
        float beta;
    };
    static size_t getBrgIdx(size_t kIdx, size_t nIdx);

    static void initBrgemmXsmm(brgemmCtx& ctx, libxsmm_xmmfunction *brgKernel, libxsmm_xmmfunction *brgKernelTileCfg, bool use_amx);
    static libxsmm_datatype dnnl_to_xsmm_dtype(dnnl_data_type_t dnnl_dtype);
    void emit_brgemm_kernel_call_libxsmm(const libxsmm_xmmfunction *xsmm_func, const libxsmm_xmmfunction *xsmm_tile_cfg, const brgemmCtx& ctx,
                                 Xbyak::Reg64 addr_A, Xbyak::Reg64 addr_B, Xbyak::Reg64 scratch, Xbyak::Reg64 addr_C,
                                 size_t in0_kernel_offset = 0, size_t in1_kernel_offset = 0,
                                 size_t in2_kernel_offset = 0, size_t out0_kernel_offset = 0) const;
    static void kernel_execute_libxsmm(libxsmm_xmmfunction *brg_kernel, void *A, void *B, void *C);
    static void libxsmm_amx_tile_configure(libxsmm_xmmfunction *cfg_kernel);
    void emit_N_blocking_loops(size_t k_kernel_id,
                               const Xbyak::Reg64& input_0, const Xbyak::Reg64& input_1,
                               const Xbyak::Reg64& input_2, const Xbyak::Reg64& output_0,
                               const Xbyak::Reg64& work_amount_N) const;

    // Note: K dimension is covered by TWO blocked kernels (with beta = 0 and 1) + 1 for tail
    static constexpr size_t BRGEMM_K_KERNEL_NUM = 3;
    static constexpr size_t BRGEMM_N_KERNEL_NUM = 2;
    std::array<brgemmCtx, BRGEMM_K_KERNEL_NUM * BRGEMM_N_KERNEL_NUM> m_brgCtxs;

    std::array<libxsmm_xmmfunction, BRGEMM_K_KERNEL_NUM * BRGEMM_N_KERNEL_NUM> m_brgKernelsXsmm;
    std::array<libxsmm_xmmfunction, BRGEMM_K_KERNEL_NUM * BRGEMM_N_KERNEL_NUM> m_brgKernelsXsmmTileCfg;

    size_t m_M;
    size_t m_K, m_K_blk, m_K_tail;
    size_t m_N, m_N_blk, m_N_tail;
    size_t m_brg0VnniFactor;
    bool m_N_blk_loop = false;
    bool m_K_blk_loop = false;

    bool m_with_scratch = false;
    bool m_with_comp = false;

    size_t m_load_offset_a = 0lu;
    size_t m_load_offset_b = 0lu;
    size_t m_load_offset_scratch = 0lu;
    size_t m_store_offset_c = 0lu;

    std::vector<size_t> io_data_size {};
};

class BinaryEltwiseTppEmitter : public jit_emitter {
public:
    BinaryEltwiseTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
    size_t get_inputs_num() const override { return 2; }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);
    static  libxsmm_blasint get_broadcasted_dim(libxsmm_blasint dim0, libxsmm_blasint dim1, std::pair<bool, bool>& bcast_flags);
//    size_t aux_gprs_count() const override;
    static libxsmm_datatype ov_to_xsmm_dtype(ov::element::Type_t elemet_type);

private:
    using jit_emitter::emit_code;
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    static void execute_libxsmm_kernel(libxsmm_meltwfunction_binary *eltwise_kernel, void *in0, void *in1, void *out0);
    libxsmm_meltwfunction_binary libxsmm_kernel{nullptr};
};

}   // namespace intel_cpu
}   // namespace ov
