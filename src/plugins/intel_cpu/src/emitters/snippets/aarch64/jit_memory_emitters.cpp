// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_memory_emitters.hpp"

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/plugin/aarch64/jit_load_store_emitters.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/store.hpp"
#include "utils/general_utils.h"

using namespace Xbyak_aarch64;

#define GET_OFF(field) offsetof(jit_snippets_call_args, field)

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_memory_emitter::jit_memory_emitter(jit_generator* h,
                                       cpu_isa_t isa,
                                       const ExpressionPtr& expr,
                                       emitter_in_out_map in_out_type)
    : jit_emitter(h, isa) {
    in_out_type_ = in_out_type;

    const auto n = expr->get_node();
    src_prc = n->get_input_element_type(0);
    dst_prc = n->get_output_element_type(0);

    const auto& memory_access = std::dynamic_pointer_cast<ov::snippets::modifier::MemoryAccess>(expr->get_node());
    if (in_out_type_ == emitter_in_out_map::gpr_to_vec) {
        OV_CPU_JIT_EMITTER_ASSERT(memory_access->is_memory_access_input_port(0), "must be input port - memory access");
        count = memory_access->get_input_count();
        compiled_byte_offset = memory_access->get_input_offset();
        buffer_cluster_id = get_parent_buffer_cluster_id(expr);
    } else if (in_out_type_ == emitter_in_out_map::vec_to_gpr) {
        OV_CPU_JIT_EMITTER_ASSERT(memory_access->is_memory_access_output_port(0),
                                  "must be output port - memory access");
        count = memory_access->get_output_count();
        compiled_byte_offset = memory_access->get_output_offset();
        buffer_cluster_id = get_consumer_buffer_cluster_id(expr);
    } else {
        std::cout << "in_out_type: " << in_out_type_ << std::endl;
        OV_CPU_JIT_EMITTER_THROW("unsupported in_out_type");
    }

    if (ov::snippets::utils::is_dynamic_value(compiled_byte_offset)) {
        is_offset_runtime = true;
        // Compiled byte offset is zero to manually `add` runtime offset before operation and `sub` after to reset
        // pointer in the register
        compiled_byte_offset = 0;
        OV_CPU_JIT_EMITTER_ASSERT(buffer_cluster_id != SIZE_MAX, "Incorrect buffer offset in call_args");
    }
}

size_t jit_memory_emitter::get_parent_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr) {
    OV_CPU_JIT_EMITTER_ASSERT(expr->get_input_port_connectors().size() == 1, "MemoryAccess must have one parent");
    const auto& parent_expr = expr->get_input_port_connector(0)->get_source().get_expr();
    if (const auto buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(parent_expr)) {
        return buffer->get_cluster_id();
    }
    return SIZE_MAX;
}

size_t jit_memory_emitter::get_consumer_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr) {
    OV_CPU_JIT_EMITTER_ASSERT(expr->get_output_port_connectors().size() == 1, "MemoryAccess must have one consumer");
    const auto& consumers = expr->get_output_port_connector(0)->get_consumers();
    for (const auto& consumer : consumers) {
        if (const auto buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(consumer.get_expr())) {
            return buffer->get_cluster_id();
        }
    }
    return SIZE_MAX;
}

jit_load_memory_emitter::jit_load_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::gpr_to_vec) {
    bool is_supported_precision =
        one_of(src_prc, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8) &&
        src_prc == dst_prc;
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");

    const auto load = ov::as_type_ptr<snippets::op::Load>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(load != nullptr, "Expects Load expression");
    count = load->get_count();
    load_emitter = std::make_unique<jit_load_emitter>(h, isa, src_prc, dst_prc, count, compiled_byte_offset);
}

void jit_load_memory_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_memory_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(load_emitter != nullptr, "Load CPU emitter isn't initialized!");

    if (is_offset_runtime) {
        XReg aux_reg(aux_gpr_idxs.back());
        XReg base_reg(in[0]);
        XReg reg_runtime_params = XReg(Operand::X0);
        // load the runtime offset from args.buffer_offsets[buffer_cluster_id]
        h->ldr(aux_reg,
               ptr(reg_runtime_params,
                   static_cast<int32_t>(GET_OFF(buffer_offsets) + buffer_cluster_id * sizeof(size_t))));
        // bump the pointer
        h->add(base_reg, base_reg, aux_reg);
    }

    load_emitter->emit_code(in, out, aux_vec_idxs, aux_gpr_idxs);

    if (is_offset_runtime) {
        XReg aux_reg(aux_gpr_idxs.back());
        XReg base_reg(in[0]);
        // subtract back so we leave the pointer unchanged for the caller
        h->sub(base_reg, base_reg, aux_reg);
    }
}

void jit_load_memory_emitter::emit_data() const {
    load_emitter->emit_data();
}

jit_load_broadcast_emitter::jit_load_broadcast_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::gpr_to_vec) {
    OV_CPU_JIT_EMITTER_ASSERT(src_prc == dst_prc,
                              "Only support equal input and output types but gets ",
                              src_prc.get_type_name(),
                              " and ",
                              dst_prc.get_type_name());
    OV_CPU_JIT_EMITTER_ASSERT(src_prc == ov::element::f32, "Only supports FP32 precision.");

    const auto broadcast_load = ov::as_type_ptr<snippets::op::BroadcastLoad>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(broadcast_load != nullptr, "Expects BroadcastLoad expression");
}

void jit_load_broadcast_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_broadcast_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = XReg(in[0]);
    auto dst = TReg(out[0]);

    h->uni_ld1rw(dst.s, src, compiled_byte_offset);
}

jit_store_memory_emitter::jit_store_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::vec_to_gpr) {
    bool is_supported_precision =
        one_of(dst_prc, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8) &&
        src_prc == dst_prc;
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");

    const auto store = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(store != nullptr, "Expects Store expression");
    count = store->get_count();
    store_emitter = std::make_unique<jit_store_emitter>(h, isa, src_prc, dst_prc, count, compiled_byte_offset);
}

void jit_store_memory_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_store_memory_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(store_emitter != nullptr, "Store CPU emitter isn't initialized!");

    if (is_offset_runtime) {
        XReg aux_reg(aux_gpr_idxs.back());
        XReg base_reg(out[0]);
        // load the runtime offset from args.buffer_offsets[buffer_cluster_id]
        XReg reg_runtime_params = XReg(Operand::X0);
        h->ldr(aux_reg,
               ptr(reg_runtime_params,
                   static_cast<int32_t>(GET_OFF(buffer_offsets) + buffer_cluster_id * sizeof(size_t))));
        // bump the pointer
        h->add(base_reg, base_reg, aux_reg);
    }

    store_emitter->emit_code(in, out, aux_vec_idxs, aux_gpr_idxs);

    if (is_offset_runtime) {
        XReg aux_reg(aux_gpr_idxs.back());
        XReg base_reg(out[0]);
        // subtract back so we leave the pointer unchanged
        h->sub(base_reg, base_reg, aux_reg);
    }
}

void jit_store_memory_emitter::emit_data() const {
    store_emitter->emit_data();
}

}  // namespace ov::intel_cpu::aarch64
