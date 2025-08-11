// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_memory_emitters.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <common/utils.hpp>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/plugin/aarch64/jit_load_store_emitters.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/store.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

using namespace Xbyak_aarch64;

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
    OV_CPU_JIT_EMITTER_ASSERT(expr->get_input_count() == 1, "MemoryAccess must have one parent");
    const auto& parent_expr = expr->get_input_port_connector(0)->get_source().get_expr();
    if (const auto buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(parent_expr)) {
        return buffer->get_cluster_id();
    }
    return SIZE_MAX;
}

size_t jit_memory_emitter::get_consumer_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr) {
    OV_CPU_JIT_EMITTER_ASSERT(expr->get_output_count() == 1, "MemoryAccess must have one output");
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
        any_of(src_prc, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8) &&
        src_prc == dst_prc;
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");

    const auto load = ov::as_type_ptr<snippets::op::Load>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(load != nullptr, "Expects Load expression");
    count = load->get_count();
    load_emitter = std::make_unique<jit_load_emitter>(h, isa, src_prc, dst_prc, count, compiled_byte_offset);
}

size_t jit_memory_emitter::get_aux_gprs_count() const {
    // for runtime arguments
    return is_offset_runtime ? 1 : 0;
}

std::vector<size_t> jit_memory_emitter::get_available_aux_gprs() const {
    OV_CPU_JIT_EMITTER_ASSERT(IMPLICATION(is_offset_runtime, !aux_gpr_idxs.empty()),
                              "If offset is dynamic, memory emitter need to have one aux gpr at least!");
    auto available_aux_gprs = aux_gpr_idxs;
    if (is_offset_runtime) {
        available_aux_gprs.pop_back();
    }
    return available_aux_gprs;
}

void jit_memory_emitter::emit_code_impl(const std::vector<size_t>& in_idxs,
                                        const std::vector<size_t>& out_idxs,
                                        const std::vector<size_t>& pool_vec_idxs,
                                        const std::vector<size_t>& pool_gpr_idxs) const {
    emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);

    auto reg_runtime_params = dnnl::impl::cpu::aarch64::abi_param1;
    XReg aux_gpr = is_offset_runtime ? XReg(static_cast<int>(aux_gpr_idxs.back())) : XReg(0);

    XReg data_reg(0);
    if (in_out_type_ == emitter_in_out_map::gpr_to_vec) {
        data_reg = XReg(in_idxs[0]);
    } else if (in_out_type_ == emitter_in_out_map::vec_to_gpr) {
        data_reg = XReg(out_idxs[0]);
    } else {
        OV_CPU_JIT_EMITTER_THROW("unsupported in_out_type");
    }

    if (is_offset_runtime) {
        // load the runtime offset from args.buffer_offsets[buffer_cluster_id]
        h->ldr(aux_gpr,
               ptr(reg_runtime_params,
                   static_cast<int32_t>(GET_OFF(buffer_offsets) + buffer_cluster_id * sizeof(size_t))));
        // bump the pointer
        h->add(data_reg, data_reg, aux_gpr);
    }

    emit_impl(in_idxs, out_idxs);

    if (is_offset_runtime) {
        // subtract back so we leave the pointer unchanged for the caller
        h->sub(data_reg, data_reg, aux_gpr);
    }

    emitter_postamble();
}

void jit_load_memory_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(load_emitter != nullptr, "Load CPU emitter isn't initialized!");
    load_emitter->emit_code(in, out, aux_vec_idxs, get_available_aux_gprs());
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
    OV_CPU_JIT_EMITTER_ASSERT(any_of(src_prc.size(), 1U, 2U, 4U), "Unsupported element type: ", src_prc);

    byte_size = src_prc.size();

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

    auto load_broadcast = [&](auto reg_view) {
        if (compiled_byte_offset == 0) {
            h->ld1r(reg_view, ptr(src));
        } else {
            h->add_imm(h->X_DEFAULT_ADDR, src, compiled_byte_offset, h->X_TMP_0);
            h->ld1r(reg_view, ptr(h->X_DEFAULT_ADDR));
        }
    };

    switch (byte_size) {
    case 1:
        load_broadcast(dst.b);
        break;
    case 2:
        load_broadcast(dst.h);
        break;
    case 4:
        h->uni_ld1rw(dst.s, src, compiled_byte_offset);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported data size ", byte_size);
    }
}

jit_store_memory_emitter::jit_store_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::vec_to_gpr) {
    bool is_supported_precision =
        any_of(dst_prc, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8) &&
        src_prc == dst_prc;
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");

    const auto store = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(store != nullptr, "Expects Store expression");
    count = store->get_count();
    store_emitter = std::make_unique<jit_store_emitter>(h, isa, src_prc, dst_prc, count, compiled_byte_offset);
}

void jit_store_memory_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(store_emitter != nullptr, "Store CPU emitter isn't initialized!");
    store_emitter->emit_code(in, out, aux_vec_idxs, get_available_aux_gprs());
}

void jit_store_memory_emitter::emit_data() const {
    store_emitter->emit_data();
}

}  // namespace ov::intel_cpu::aarch64
