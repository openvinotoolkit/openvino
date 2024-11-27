// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_equation_emitter.hpp"
#include "transformations/tpp/x64/op/equation.hpp"
#include "emitters/plugin/x64/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

EquationTppEmitter::EquationTppEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) :
                                       TppEmitter(h, isa, expr), m_num_inputs(expr->get_input_count()) {
    const auto& eq_tpp = ov::as_type_ptr<tpp::op::EquationTPP>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(eq_tpp, "Invalid TPP node type detected");
    auto get_MN = [this](int arg_idx){
        const auto& subtensor = get_projected_subtensor(io_port_descriptors[arg_idx]);
        OV_CPU_JIT_EMITTER_ASSERT(subtensor.size() == 2, "TPP supports only 2D subtensors");
        return std::make_pair(static_cast<libxsmm_blasint>(*++subtensor.rbegin()),
                              static_cast<libxsmm_blasint>(*subtensor.rbegin()));
    };

    m_equation_id = libxsmm_meqn_create();
    const auto op_metadata = libxsmm_create_meqn_op_metadata(m_equation_id, -1);
    const auto sing_attr = libxsmm_create_matrix_arg_attributes(LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);
    libxsmm_blasint M, N;
    for (const auto& op_desc : eq_tpp->get_op_descs()) {
        switch (op_desc.get_arity()) {
            case tpp::op::OpDescTPP::ARITY::BINARY: {
                auto flags = op_desc.get_flags();
                libxsmm_meqn_push_back_binary_op(op_metadata, op_desc, exec_dtype, flags);
                break;
            } case tpp::op::OpDescTPP::ARITY::UNARY: {
                libxsmm_meqn_push_back_unary_op(op_metadata, op_desc, exec_dtype, LIBXSMM_MELTW_FLAG_UNARY_NONE);
                break;
            } case tpp::op::OpDescTPP::ARITY::ZERO: {
                const auto arg_idx = static_cast<int>(op_desc);
                std::tie(M, N) = get_MN(arg_idx);
                auto metadata = libxsmm_create_meqn_arg_metadata(m_equation_id, arg_idx);
                auto shape = libxsmm_create_meqn_arg_shape(N, M,
                                                           static_cast<libxsmm_blasint>(io_strides[arg_idx]),
                                                           io_dtypes[arg_idx]);
                OV_CPU_JIT_EMITTER_ASSERT(libxsmm_meqn_push_back_arg(metadata, shape, sing_attr) == 0,
                                          "Failed to push back arg to tpp equation");
                break;
            }
            default:
                OV_CPU_JIT_EMITTER_THROW("Unhandled tpp::op::OpDescTPP::ARITY");
        }
    }
    // Note: for debug purposes it might be useful to serialize the equations graph here
    // libxsmm_meqn_tree_print(m_equation_id);
    // libxsmm_meqn_rpn_print(m_equation_id);
    std::tie(M, N) = get_MN(static_cast<int>(io_port_descriptors.size()) - 1);
    m_out_shape = libxsmm_create_meqn_arg_shape(N, M,
                                                static_cast<libxsmm_blasint>(io_strides.back()),
                                                io_dtypes.back());
}

size_t EquationTppEmitter::get_inputs_num() const {
    return m_num_inputs;
}

const uintptr_t EquationTppEmitter::get_compiled_kernel_ptr() const {
    return COMPILE_TPP_KERNEL(libxsmm_dispatch_meqn(m_equation_id, m_out_shape));
}

std::set<std::vector<element::Type>> EquationTppEmitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    // Note: TPPs have build-in convert semantics, so the equations should support any input precision (specified when created)
    OV_CPU_JIT_EMITTER_ASSERT(node && ov::is_type<tpp::op::EquationTPP>(node), "Invalid node ptr or type");
    std::vector<element::Type> input_precs;
    for (const auto& in : node->inputs())
        input_precs.push_back(in.get_element_type());
    return {input_precs};
}


void EquationTppEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == m_num_inputs, "Expects " + std::to_string(m_num_inputs) +
                              " input registers, got " + std::to_string(in.size()));
    const auto num_outputs = num_kernel_args - m_num_inputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == num_outputs, "Expects " + std::to_string(num_outputs) +
                              " output register, got " + std::to_string(out.size()));
}

void EquationTppEmitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    EmitABIRegSpills spill(h);
    spill.preamble();

    // save function address in gpr to pass in call instruction
    h->mov(h->rbp, get_execute_function_ptr());

    // Reserve memory on the stack
    h->sub(h->rsp, num_kernel_args * sizeof(void*));
    // Write data ptr registers content + apply offsets
    for (size_t i = 0; i < static_cast<size_t>(num_kernel_args); i++) {
        auto reg_idx = i < in.size() ? in[i] : out[i - in.size()];
        const auto addr = h->rsp + i * sizeof(void*);
        h->mov(h->qword[addr], Reg64(static_cast<int>(reg_idx)));
        const auto bytes_offset = io_offsets[i];
        if (bytes_offset)
            h->add(h->qword[addr], bytes_offset);
    }

    const auto& compiled_kernel = get_compiled_kernel_ptr();
    OV_CPU_JIT_EMITTER_ASSERT(compiled_kernel, "Failed to compile libxsmm_kernel");

    // Pass arguments according to the execute signature
    h->mov(abi_param1, compiled_kernel);
    h->mov(abi_param2, num_kernel_args);
    h->mov(abi_param3, h->rsp);

    spill.rsp_align();
    h->call(h->rbp);
    spill.rsp_restore();

    // Free allocated memory on the stack
    h->add(h->rsp, num_kernel_args * sizeof(void*));
    spill.postamble();
}

void EquationTppEmitter::execute_kernel(libxsmm_meqn_function equation_kernel, int argc, void **argv) {
    std::vector<libxsmm_matrix_arg> inputs(argc - 1);
    for (int i = 0; i < argc - 1; i++)
        inputs[i].primary = argv[i];
    libxsmm_meqn_param param;
    param.ops_args = nullptr;
    param.inputs = inputs.data();
    param.output.primary = argv[argc-1];
    equation_kernel(&param);
}

}  // namespace intel_cpu
}  // namespace ov
