// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_tpp_emitter.hpp"

#include "emitters/plugin/x64/utils.hpp"
#include "emitters/tpp/common/utils.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "transformations/tpp/x64/op/eltwise.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

VectorDims TppEmitter::get_projected_subtensor(const snippets::lowered::PortDescriptorPtr& desc) {
    auto shape = desc->get_shape();
    auto subtensor = desc->get_subtensor();
    // Note: Scalar is a special case, so it's easier to prepend shapes than to handle it explicitly
    if (shape.size() == 1) {
        shape.insert(shape.begin(), 1);
    }
    if (subtensor.size() == 1) {
        subtensor.insert(subtensor.begin(), 1);
    }
    OV_CPU_JIT_EMITTER_ASSERT(subtensor.size() <= shape.size() && !subtensor.empty(),
                              "Invalid subtensor + shape combination");
    auto shape_it = shape.rbegin();
    for (auto sub_it = subtensor.rbegin(); sub_it != subtensor.rend(); sub_it++, shape_it++) {
        *sub_it = std::min(*sub_it, *shape_it);
    }
    return subtensor;
}

TppEmitter::TppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                       dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_binary_call_emitter(h, isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& node = expr->get_node();
    const auto& tpp_mod = std::dynamic_pointer_cast<tpp::modifier::TensorProcessingPrimitive>(node);
    OV_CPU_JIT_EMITTER_ASSERT(tpp_mod, "Invoked with invalid node type");

    const auto num_ins = node->get_input_size();
    const auto num_outs = node->get_output_size();
    num_kernel_args = static_cast<int>(num_ins + num_outs);
    io_dtypes.resize(num_kernel_args);
    io_strides.resize(num_kernel_args);
    io_offsets.resize(num_kernel_args);
    io_port_descriptors.resize(num_kernel_args);
    // Note: this is needed mostly for Reduce operations, since they allow the last subternsor dim to be FULL_DIM;
    auto replace_full_dim = [](size_t dim, size_t replace_dim) {
        if (ov::snippets::utils::is_full_dim_value(dim)) {
            return replace_dim;
        }
        return dim;
    };

    for (size_t i = 0; i < num_ins; i++) {
        io_dtypes[i] = tpp::utils::ov_to_xsmm_dtype(node->get_input_element_type(i));
        io_offsets[i] = tpp_mod->get_input_offset(i);
        io_strides[i] =
            replace_full_dim(tpp_mod->get_input_stride(i), expr->get_input_port_descriptor(i)->get_shape().back());
        io_port_descriptors[i] = expr->get_input_port_descriptor(i);
    }

    for (size_t i = 0; i < num_outs; i++) {
        const auto i_off = i + num_ins;
        io_dtypes[i_off] = tpp::utils::ov_to_xsmm_dtype(node->get_output_element_type(i));
        io_offsets[i_off] = tpp_mod->get_output_offset(i);
        io_strides[i_off] =
            replace_full_dim(tpp_mod->get_output_stride(i), expr->get_output_port_descriptor(i)->get_shape().back());
        io_port_descriptors[i_off] = expr->get_output_port_descriptor(i);
    }
}

void TppEmitter::emit_code_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void TppEmitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    // Note: 4 args is currently enough for unary and binary ops.
    // To enable ternary ops, we will have to pass extra regs on stack for Windows,
    std::array<Xbyak::Reg64, 4> abi_params{abi_param1, abi_param2, abi_param3, abi_param4};
    init_binary_call_regs(abi_params.size(), in, out);

    const Xbyak::Reg64& aux_reg = get_call_address_reg();
    const Xbyak::Reg64& callee_saved_reg = get_callee_saved_reg();

    EmitABIRegSpills spill(h);
    spill.preamble(get_regs_to_spill());

    int aux_xmm_count = 0;
    for (auto reg_idx : in)
        h->uni_vmovq(Xmm(aux_xmm_count++), Reg64(static_cast<int>(reg_idx)));
    for (auto reg_idx : out)
        h->uni_vmovq(Xmm(aux_xmm_count++), Reg64(static_cast<int>(reg_idx)));

    OV_CPU_JIT_EMITTER_ASSERT(aux_xmm_count == num_kernel_args, "offsets for some inputs/outputs were not set");
    OV_CPU_JIT_EMITTER_ASSERT(aux_xmm_count < static_cast<int>(abi_params.size()),
                              "too many input/output arguments. More abi params required");

    const auto data_ptr_reg = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) {
            h->add(reg, bytes_offset);
        }
    };
    const auto& compiled_kernel = get_compiled_kernel_ptr();
    OV_CPU_JIT_EMITTER_ASSERT(compiled_kernel, "Failed to compile libxsmm_kernel");

    h->mov(abi_params[0], compiled_kernel);
    for (int i = 0; i < num_kernel_args; i++)
        data_ptr_reg(Xmm(i), abi_params[i + 1], io_offsets[i]);
    // save function address in gpr to pass in call instruction
    h->mov(aux_reg, get_execute_function_ptr());

    spill.rsp_align(callee_saved_reg.getIdx());
    h->call(aux_reg);
    spill.rsp_restore();

    spill.postamble();
}

}  // namespace ov::intel_cpu
