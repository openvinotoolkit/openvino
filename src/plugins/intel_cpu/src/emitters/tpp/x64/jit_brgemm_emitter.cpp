// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"
#include "emitters/snippets/x64/jit_snippets_emitters.hpp"
#include "transformations/tpp/x64/op/brgemm.hpp"

using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

namespace ov {
namespace intel_cpu {

void BrgemmTppEmitter::validate_subtensors(const VectorDims& in_0, const VectorDims& in_1, const VectorDims& out_0) {
    bool subtensors_compatible = in_0.size() == in_1.size() && in_0.size() == out_0.size() && in_0.size() == 2 &&
                                 in_0[1] == in_1[0] && in_0[0] == out_0[0] && in_1[1] == out_0[1];
    OV_CPU_JIT_EMITTER_ASSERT(subtensors_compatible, "Incompatible subtensors");
}

BrgemmTppEmitter::BrgemmTppEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : TppEmitter(h, isa, expr) {
    const auto& brgemm_node = as_type_ptr<intel_cpu::tpp::op::BrgemmTPP>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_node && !brgemm_node->is_dynamic(), "Invoked with invalid node type");

    const auto& input_0_desc = expr->get_input_port_descriptor(0);
    const auto& input_1_desc = expr->get_input_port_descriptor(1);
    const auto& output_desc = expr->get_output_port_descriptor(0);

    std::vector<size_t> leading_dimensions {brgemm_node->get_input_stride(0),
                                            brgemm_node->get_input_stride(1),
                                            brgemm_node->get_output_stride(0)};

    auto in_0_prec = ov_to_xsmm_dtype(brgemm_node->get_input_element_type(0));
    auto in_1_prec = ov_to_xsmm_dtype(brgemm_node->get_input_element_type(1));
    exec_dtype = in_0_prec == LIBXSMM_DATATYPE_I8 || in_0_prec == LIBXSMM_DATATYPE_U8 ?
                  LIBXSMM_DATATYPE_I32 :
                  LIBXSMM_DATATYPE_F32;
    auto out_0_prec = exec_dtype == LIBXSMM_DATATYPE_I32 ?
                      LIBXSMM_DATATYPE_I32 :
                      LIBXSMM_DATATYPE_F32;

    const auto beta = brgemm_node->get_beta();
    OV_CPU_JIT_EMITTER_ASSERT(beta == 0 || beta == 1, "Detected unsupported beta value: " + std::to_string(beta));

    const auto& subtensor_in0 = input_0_desc->get_subtensor();
    const auto& subtensor_in1 = input_1_desc->get_subtensor();
    const auto& subtensor_out0 = output_desc->get_subtensor();
    validate_subtensors(subtensor_in0, subtensor_in1, subtensor_out0);

    const auto K = static_cast<libxsmm_blasint>(*subtensor_in0.rbegin());
    const auto M = static_cast<libxsmm_blasint>(*++subtensor_in0.rbegin());
    const auto N = static_cast<libxsmm_blasint>(*subtensor_in1.rbegin());

    const bool is_f32_gemm = in_0_prec == in_1_prec && in_0_prec == LIBXSMM_DATATYPE_F32;
    const bool is_bf16_gemm =  in_0_prec == in_1_prec && in_0_prec == LIBXSMM_DATATYPE_BF16;
    const bool is_i8_gemm = in_0_prec == LIBXSMM_DATATYPE_U8 || in_0_prec == LIBXSMM_DATATYPE_I8;
    OV_CPU_JIT_EMITTER_ASSERT(is_f32_gemm ||
                              (is_bf16_gemm && K % 2 == 0) ||
                              (is_i8_gemm && K % 4 == 0),
                              "Unsupported parameter combination for kernel configuration");

    m_compile_flags = is_f32_gemm ?
                      LIBXSMM_GEMM_FLAGS('N', 'N') :
                      LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') |
                      LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
                      LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;

    if (beta == 0)
        m_compile_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

    if (in_0_prec == LIBXSMM_DATATYPE_U8) {
        in_0_prec = LIBXSMM_DATATYPE_I8;
        m_compile_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED;
    }
    if (in_1_prec == LIBXSMM_DATATYPE_U8) {
        in_1_prec = LIBXSMM_DATATYPE_I8;
        m_compile_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED;
    }

    m_shape = libxsmm_create_gemm_shape(N, M, K,
                                        io_strides[1], io_strides[0], io_strides[2],
                                        in_1_prec, in_0_prec, out_0_prec,
                                        exec_dtype);
    m_prefetching_flags = LIBXSMM_GEMM_PREFETCH_NONE;
}

std::set<std::vector<element::Type>> BrgemmTppEmitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    // Note: BrgemmTpp currently supports only fp32
    return {{element::f32, element::f32}};
}

void BrgemmTppEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 2, "Expects 2 input regs, got" + std::to_string(in.size()));
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got" + std::to_string(out.size()));
}

const uintptr_t BrgemmTppEmitter::get_compiled_kernel_ptr() const {
    return COMPILE_TPP_KERNEL(libxsmm_dispatch_gemm(m_shape, m_compile_flags, m_prefetching_flags));
}

void BrgemmTppEmitter::execute_brgemm_kernel(libxsmm_gemmfunction brg_kernel, void *in0, void *in1, void *out0) {
    libxsmm_gemm_param gemm_p;
    gemm_p.a.primary = in1;
    gemm_p.b.primary = in0;
    gemm_p.c.primary = out0;
    OV_CPU_JIT_EMITTER_ASSERT(brg_kernel, "Invalid brgemm kernel pointer");
    brg_kernel(&gemm_p);
}

}  // namespace intel_cpu
}  // namespace ov
