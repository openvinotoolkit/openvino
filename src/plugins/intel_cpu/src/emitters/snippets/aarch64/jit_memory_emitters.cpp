// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_memory_emitters.hpp"
#include "transformations/snippets/common/op/load_convert.hpp"
#include "transformations/snippets/common/op/store_convert.hpp"
#include "emitters/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_memory_emitter::jit_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr, emitter_in_out_map in_out_type)
    : jit_emitter(h, isa) {
    in_out_type_ = in_out_type;

    const auto n = expr->get_node();
    src_prc = n->get_input_element_type(0);
    dst_prc = n->get_output_element_type(0);

    const auto& memory_access = std::dynamic_pointer_cast<ov::snippets::modifier::MemoryAccess>(expr->get_node());
    if (in_out_type_ == emitter_in_out_map::gpr_to_vec) {
        OV_CPU_JIT_EMITTER_ASSERT(memory_access->is_memory_access_input_port(0), "Must be input port - memory access");
        count = memory_access->get_input_count();
        byte_offset = memory_access->get_input_offset();
    } else if (in_out_type_ == emitter_in_out_map::vec_to_gpr) {
        OV_CPU_JIT_EMITTER_ASSERT(memory_access->is_memory_access_output_port(0), "Must be output port - memory access");
        count = memory_access->get_output_count();
        byte_offset = memory_access->get_output_offset();
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported in_out_type");
    }
}

jit_load_memory_emitter::jit_load_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::gpr_to_vec) {
    bool is_supported_precision = one_of(src_prc, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8) &&
                                  (src_prc == dst_prc || one_of(dst_prc, ov::element::f32, ov::element::i32));
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");

    const auto load = std::dynamic_pointer_cast<snippets::op::Load>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(load != nullptr, "Expects Load expression");
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count, byte_offset));
}

void jit_load_memory_emitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_memory_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(load_emitter != nullptr, "Load CPU emitter isn't initialized!");

    load_emitter->emit_code(in, out, aux_vec_idxs, aux_gpr_idxs);
}

void jit_load_memory_emitter::emit_data() const {
    load_emitter->emit_data();
}

jit_load_broadcast_emitter::jit_load_broadcast_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::gpr_to_vec) {
    OV_CPU_JIT_EMITTER_ASSERT(src_prc == dst_prc, "Only support equal input and output types but gets ",
                              src_prc.get_type_name(),
                              " and ",
                              dst_prc.get_type_name());
    OV_CPU_JIT_EMITTER_ASSERT(src_prc == ov::element::f32, "Only supports FP32 precision.");

    const auto broadcast_load = std::dynamic_pointer_cast<snippets::op::BroadcastLoad>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(broadcast_load != nullptr, "Expects BroadcastLoad expression");
}

void jit_load_broadcast_emitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_broadcast_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    XReg src = XReg(in[0]);
    TReg dst = TReg(out[0]);

    h->uni_ld1rw(dst.s, src, byte_offset);
}

jit_store_memory_emitter::jit_store_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::vec_to_gpr) {
    bool is_supported_precision = one_of(dst_prc, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8) &&
                                  (src_prc == dst_prc || one_of(src_prc, ov::element::f32, ov::element::i32));
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");

    if (ov::is_type<ov::intel_cpu::StoreConvertTruncation>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, byte_offset, arithmetic_mode::truncation));
    } else if (ov::is_type<ov::intel_cpu::StoreConvertSaturation>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, byte_offset, arithmetic_mode::saturation));
    } else if (ov::is_type<ov::snippets::op::Store>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, byte_offset));
    } else {
        OV_CPU_JIT_EMITTER_THROW("Expects Store node");
    }
}

void jit_store_memory_emitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_store_memory_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(store_emitter != nullptr, "Store CPU emitter isn't initialized!");

    store_emitter->emit_code(in, out, aux_vec_idxs, aux_gpr_idxs);
}

void jit_store_memory_emitter::emit_data() const {
    store_emitter->emit_data();
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
