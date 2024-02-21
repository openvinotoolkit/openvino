// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"
#include "transformations/snippets/tpp/op/eltwise.hpp"

namespace ov {
namespace intel_cpu {
using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

BinaryEltwiseTppEmitter::BinaryEltwiseTppEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) :
                                                 TppEmitter(h, isa, expr) {
    const auto& subtensor_in0 = get_projected_subtensor(io_port_descriptors[0]);
    const auto& subtensor_in1 = get_projected_subtensor(io_port_descriptors[1]);

    const auto N_in0 = static_cast<libxsmm_blasint>(*subtensor_in0.rbegin());
    const auto M_in0 = static_cast<libxsmm_blasint>(*++subtensor_in0.rbegin());
    const auto N_in1 = static_cast<libxsmm_blasint>(*subtensor_in1.rbegin());
    const auto M_in1 = static_cast<libxsmm_blasint>(*++subtensor_in1.rbegin());

    std::pair<bool, bool> n_bcast_flags, m_bcast_flags;
    const auto N = get_broadcasted_dim(N_in0, N_in1, n_bcast_flags);
    const auto M = get_broadcasted_dim(M_in0, M_in1, m_bcast_flags);

    m_compile_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
    if (m_bcast_flags.first && n_bcast_flags.first) {
        m_compile_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
    } else if (m_bcast_flags.first) {
        m_compile_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    } else  if (n_bcast_flags.first) {
        m_compile_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
    }
    if (m_bcast_flags.second && n_bcast_flags.second) {
        m_compile_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    } else if (m_bcast_flags.second) {
        m_compile_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
    } else  if (n_bcast_flags.second) {
        m_compile_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
    }
    const auto& binary_eltw_tpp = std::dynamic_pointer_cast<tpp::op::BinaryEltwiseTPP>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(binary_eltw_tpp, "Invalid TPP node type detected");
    m_op_type = binary_eltw_tpp->get_op_type();
    // Note: libxsmm implies column-major layout, so we have to swap M and N here
    m_shape = libxsmm_create_meltw_binary_shape(N, M,
                                                io_strides[0], io_strides[1], io_strides[2],
                                                io_dtypes[0], io_dtypes[1], io_dtypes[2],
                                                exec_dtype);
}

const uintptr_t BinaryEltwiseTppEmitter::get_compiled_kernel_ptr() const {
    // Note: libxsmm hides memory management from the user, so we don't have to store pointer to compiled kernel to keep it alive.
    // libxsmm will keep the pointer alive until the end of program execution (it doesn't matter whether we save the pointer in the emitter or not)
    return reinterpret_cast<const uintptr_t>(libxsmm_dispatch_meltw_binary(m_op_type, m_shape, m_compile_flags));
}

std::set<std::vector<element::Type>> BinaryEltwiseTppEmitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}


void BinaryEltwiseTppEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 2, "Expects 2 input registers, got " + std::to_string(in.size()));
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output register, got " + std::to_string(out.size()));
}

void BinaryEltwiseTppEmitter::execute_kernel(libxsmm_meltwfunction_binary eltwise_kernel, void *in0, void *in1, void *out0) {
    libxsmm_meltw_binary_param param;
    param.op.primary = nullptr;
    param.in0.primary = in0;
    param.in1.primary = in1;
    param.out.primary = out0;
    eltwise_kernel(&param);
}

libxsmm_blasint BinaryEltwiseTppEmitter::get_broadcasted_dim(libxsmm_blasint dim0, libxsmm_blasint dim1, std::pair<bool, bool>& bcast_flags) {
    if (dim0 == dim1) {
        bcast_flags = {false, false};
        return dim0;
    } else if (dim1 == 1) {
        bcast_flags = {false, true};
        return dim0;
    } else if (dim0 == 1) {
        bcast_flags = {true, false};
        return dim1;
    }
    OV_CPU_JIT_EMITTER_THROW("Invalid dimensions passed to get_broadcast_flags");
    return -1;
}

UnaryEltwiseTppEmitter::UnaryEltwiseTppEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) :
                                               TppEmitter(h, isa, expr) {
    const auto& subtensor_in0 = get_projected_subtensor(io_port_descriptors[0]);

    const auto N = static_cast<libxsmm_blasint>(*subtensor_in0.rbegin());
    const auto M = static_cast<libxsmm_blasint>(*++subtensor_in0.rbegin());

    const auto& unary_eltw_tpp = std::dynamic_pointer_cast<tpp::op::UnaryEltwiseTPP>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(unary_eltw_tpp, "Invalid TPP node type detected");
    m_op_type = unary_eltw_tpp->get_op_type();
    // Note: libxsmm implies column-major layout, so we have to swap M and N here
    m_shape = libxsmm_create_meltw_unary_shape(N, M,
                                               io_strides[0], io_strides[1],
                                               io_dtypes[0], io_dtypes[1],
                                               exec_dtype);
    m_compile_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
}

void UnaryEltwiseTppEmitter::execute_kernel(libxsmm_meltwfunction_unary eltwise_kernel, void *in0, void *out0) {
    libxsmm_meltw_unary_param param;
    param.op.primary = nullptr;
    param.in.primary = in0;
    param.out.primary = out0;
    eltwise_kernel(&param);
}

std::set<std::vector<element::Type>> UnaryEltwiseTppEmitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void UnaryEltwiseTppEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 1, "Expects 1 input registers, got " + std::to_string(in.size()));
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output register, got " + std::to_string(out.size()));
}

ReduceTppEmitter::ReduceTppEmitter(jit_generator* h, cpu_isa_t isa,  const ExpressionPtr& expr) :
                                   UnaryEltwiseTppEmitter(h, isa, expr) {
    m_compile_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
    // No need to set ldo for reduce, it is always assumed = 1 inside the kernel
    // m_shape.ldo = 1;
}

}  // namespace intel_cpu
}  // namespace ov
