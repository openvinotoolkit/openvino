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

    size_t get_inputs_num() const override { return 2; }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

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

    static void initBrgemmXsmm(brgemmCtx& ctx, libxsmm_gemmfunction& brgKernel, libxsmm_gemmfunction& brgKernelTileCfg, bool use_amx);
    static libxsmm_datatype dnnl_to_xsmm_dtype(dnnl_data_type_t dnnl_dtype);
    void emit_brgemm_kernel_call_libxsmm(Xbyak::Reg64 addr_A, Xbyak::Reg64 addr_B, Xbyak::Reg64 addr_C) const;
    static void kernel_execute_libxsmm(libxsmm_gemmfunction brg_kernel, void *A, void *B, void *C);
    static void libxsmm_amx_tile_configure(libxsmm_gemmfunction cfg_kernel);

    brgemmCtx m_brgCtx;

    libxsmm_gemmfunction m_brgKernelsXsmm{nullptr};
    libxsmm_gemmfunction m_brgKernelsXsmmTileCfg{nullptr};

    size_t m_load_offset_a = 0lu;
    size_t m_load_offset_b = 0lu;
    size_t m_store_offset_c = 0lu;

    std::vector<size_t> io_data_size {};
};

class EltwiseTppEmitter : public jit_emitter {
public:
    EltwiseTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
//    size_t aux_gprs_count() const override;
    static libxsmm_datatype ov_to_xsmm_dtype(ov::element::Type_t elemet_type);

protected:
    using jit_emitter::emit_code;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    virtual void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const  = 0;
    static ov::snippets::VectorDims get_projected_subtensor(const snippets::lowered::PortDescriptorPtr& desc);

    virtual uintptr_t get_execute_funcion_ptr() const = 0;
    virtual uintptr_t get_compiled_kernel_ptr() const = 0;

    std::vector<size_t> io_offsets{};
    std::vector<libxsmm_datatype> io_dtypes{};
    // Note: almost all emitters use fp32 for internal compulations
    const libxsmm_datatype dtype_comp {LIBXSMM_DATATYPE_F32};
    // aka leading dimensions
    std::vector<size_t> io_strides{};
    std::vector<snippets::lowered::PortDescriptorPtr> io_port_descriptors{};
    // compile flags has the same type for all eltwises, so we keep them in the base class
    libxsmm_bitfield m_compile_flags {0};
};

class BinaryEltwiseTppEmitter : public EltwiseTppEmitter {
public:
    BinaryEltwiseTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                            dnnl::impl::cpu::x64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override { return 2; }
    static void execute_binary_eltw_kernel(libxsmm_meltwfunction_binary eltwise_kernel, void *in0, void *in1, void *out0);
    uintptr_t get_execute_funcion_ptr() const override { return reinterpret_cast<uintptr_t>(execute_binary_eltw_kernel); }
    uintptr_t get_compiled_kernel_ptr() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

protected:
    libxsmm_meltw_binary_shape m_shape;
    libxsmm_meltw_binary_type m_op_type;
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    static  libxsmm_blasint get_broadcasted_dim(libxsmm_blasint dim0, libxsmm_blasint dim1, std::pair<bool, bool>& bcast_flags);
};

class UnaryEltwiseTppEmitter : public EltwiseTppEmitter {
public:
    UnaryEltwiseTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                            dnnl::impl::cpu::x64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override { return 1; }
    static void execute_unary_eltw_kernel(libxsmm_meltwfunction_unary eltwise_kernel, void *in0, void *out0);
    uintptr_t get_execute_funcion_ptr() const override { return reinterpret_cast<uintptr_t>(execute_unary_eltw_kernel); }
    uintptr_t get_compiled_kernel_ptr() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

protected:
    libxsmm_meltw_unary_shape m_shape;
    libxsmm_meltw_unary_type m_op_type;
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
};

class ReduceTppEmitter : public UnaryEltwiseTppEmitter {
public:
    ReduceTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);
};
}   // namespace intel_cpu
}   // namespace ov
