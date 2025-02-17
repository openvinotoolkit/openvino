// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_nop_emitter::jit_nop_emitter(jit_generator* h,
                                 cpu_isa_t isa,
                                 const ExpressionPtr& expr,
                                 emitter_in_out_map emitter_type)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_type;
}

jit_parameter_emitter::jit_parameter_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_nop_emitter(h, isa, expr) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

jit_result_emitter::jit_result_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_nop_emitter(h, isa, expr) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

jit_broadcast_move_emitter::jit_broadcast_move_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    if (n->get_input_element_type(0) != n->get_output_element_type(0)) {
        OV_CPU_JIT_EMITTER_THROW("supports only equal input and output types but gets: ",
                                 n->get_input_element_type(0),
                                 " and ",
                                 n->get_output_element_type(0));
    }
    byte_size = n->get_input_element_type(0).size();
}

void jit_broadcast_move_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_broadcast_move_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    using Vmm = typename dnnl::impl::utils::
        conditional3<isa == dnnl::impl::cpu::x64::sse41, Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    auto xmm_src0 = Xmm(in[0]);
    auto vmm_dst = Vmm(out[0]);

    switch (byte_size) {
    case 4:
        h->uni_vbroadcastss(vmm_dst, xmm_src0);
        break;
    case 2:
        h->vpbroadcastw(vmm_dst, xmm_src0);
        break;
    case 1:
        h->vpbroadcastb(vmm_dst, xmm_src0);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("unsupported data type");
    }
}

int32_t jit_scalar_emitter::read_value(const ov::snippets::lowered::ExpressionPtr& expr) {
    const auto n = ov::as_type_ptr<ov::op::v0::Constant>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(n, "Invalid node, expected op::v0::Constant");
    const auto& precision = n->get_output_element_type(0);
    int32_t res = INT_MIN;
    switch (precision) {
    case element::i32:
        res = n->cast_vector<int32_t>(1)[0];
        break;
    case element::f32:
        res = dnnl::impl::cpu::x64::float2int(n->cast_vector<float>(1)[0]);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("doesn't support ", precision);
    }
    return res;
}

jit_scalar_emitter::jit_scalar_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    push_arg_entry_of("scalar", read_value(expr), true);
    prepare_table();
}

void jit_scalar_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    using isa = cpu_isa_t;
    switch (host_isa_) {
    case isa::sse41:
        emit_isa<isa::sse41>(in, out);
        break;
    case isa::avx2:
        emit_isa<isa::avx2>(in, out);
        break;
    case isa::avx512_core:
        emit_isa<isa::avx512_core>(in, out);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_scalar_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    using Vmm = typename dnnl::impl::utils::
        conditional3<isa == dnnl::impl::cpu::x64::sse41, Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    auto vmm_dst = Vmm(out[0]);
    h->uni_vbroadcastss(vmm_dst, table_val("scalar"));
}

}  // namespace ov::intel_cpu
