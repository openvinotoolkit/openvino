// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_memory_emitters.hpp"

#include <common/utils.hpp>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include "snippets/op/broadcastload.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/store.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64 {

using jit_generator_t = ov::intel_cpu::riscv64::jit_generator_t;
using cpu_isa_t = ov::intel_cpu::riscv64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_memory_emitter::jit_memory_emitter(jit_generator_t* h,
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

jit_load_memory_emitter::jit_load_memory_emitter(jit_generator_t* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::gpr_to_vec) {
    bool is_supported_precision =
        any_of(src_prc, ov::element::f32, ov::element::i32, ov::element::f16) && src_prc == dst_prc;
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");

    const auto load = ov::as_type_ptr<snippets::op::Load>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(load != nullptr, "Expects Load expression");
    count = load->get_count();
    byte_size = src_prc.size();
}

size_t jit_memory_emitter::aux_gprs_count() const {
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
    std::vector<size_t> pool_fp_gpr_idxs; // Empty for now
    emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs, pool_fp_gpr_idxs);

    auto reg_runtime_params = Xbyak_riscv::a0;  // First ABI parameter register
    Xbyak_riscv::Reg aux_gpr = is_offset_runtime ? Xbyak_riscv::Reg(aux_gpr_idxs.back()) : Xbyak_riscv::zero;

    Xbyak_riscv::Reg data_reg = Xbyak_riscv::zero;
    if (in_out_type_ == emitter_in_out_map::gpr_to_vec) {
        data_reg = Xbyak_riscv::Reg(in_idxs[0]);
    } else if (in_out_type_ == emitter_in_out_map::vec_to_gpr) {
        data_reg = Xbyak_riscv::Reg(out_idxs[0]);
    } else {
        OV_CPU_JIT_EMITTER_THROW("unsupported in_out_type");
    }

    if (is_offset_runtime) {
        // load the runtime offset from args.buffer_offsets[buffer_cluster_id]
        const auto offset = GET_OFF(buffer_offsets) + buffer_cluster_id * sizeof(size_t);
        // RV64 uses 64-bit size_t
        h->ld(aux_gpr, reg_runtime_params, static_cast<int32_t>(offset));
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
    if (host_isa_ == ov::intel_cpu::riscv64::gv) {
        emit_isa<ov::intel_cpu::riscv64::gv>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_memory_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    auto src = Xbyak_riscv::Reg(in[0]);
    auto dst = Xbyak_riscv::VReg(out[0]);

    // Set vector configuration for the load (e16 for 2-byte, e32 for 4-byte)
    auto sew = (byte_size == 2) ? Xbyak_riscv::SEW::e16 : Xbyak_riscv::SEW::e32;
    h->vsetivli(Xbyak_riscv::zero, count, sew, Xbyak_riscv::LMUL::m1);

    // Load vector data from memory
    if (compiled_byte_offset == 0) {
        if (byte_size == 2) {
            h->vle16_v(dst, src);
        } else {
            h->vle32_v(dst, src);
        }
    } else {
        // Use temporary register to calculate address with offset
        auto tmp_gpr = Xbyak_riscv::Reg(aux_gpr_idxs.empty() ? Xbyak_riscv::t0.getIdx() : aux_gpr_idxs[0]);
        h->addi(tmp_gpr, src, static_cast<int32_t>(compiled_byte_offset));
        if (byte_size == 2) {
            h->vle16_v(dst, tmp_gpr);
        } else {
            h->vle32_v(dst, tmp_gpr);
        }
    }
}

void jit_load_memory_emitter::emit_data() const {
    // No additional data emission needed for basic load
}

jit_store_memory_emitter::jit_store_memory_emitter(jit_generator_t* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::vec_to_gpr) {
    bool is_supported_precision =
        any_of(dst_prc, ov::element::f32, ov::element::i32, ov::element::f16) && src_prc == dst_prc;
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");

    const auto store = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(store != nullptr, "Expects Store expression");
    count = store->get_count();
    byte_size = dst_prc.size();
}

void jit_store_memory_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == ov::intel_cpu::riscv64::gv) {
        emit_isa<ov::intel_cpu::riscv64::gv>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_store_memory_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    auto src = Xbyak_riscv::VReg(in[0]);
    auto dst = Xbyak_riscv::Reg(out[0]);

    // Set vector configuration for the store
    auto sew = (byte_size == 2) ? Xbyak_riscv::SEW::e16 : Xbyak_riscv::SEW::e32;
    h->vsetivli(Xbyak_riscv::zero, count, sew, Xbyak_riscv::LMUL::m1);

    // Store vector data to memory
    if (compiled_byte_offset == 0) {
        if (byte_size == 2) {
            h->vse16_v(src, dst);
        } else {
            h->vse32_v(src, dst);
        }
    } else {
        // Use temporary register to calculate address with offset
        auto tmp_gpr = Xbyak_riscv::Reg(aux_gpr_idxs.empty() ? Xbyak_riscv::t0.getIdx() : aux_gpr_idxs[0]);
        h->addi(tmp_gpr, dst, static_cast<int32_t>(compiled_byte_offset));
        if (byte_size == 2) {
            h->vse16_v(src, tmp_gpr);
        } else {
            h->vse32_v(src, tmp_gpr);
        }
    }
}

void jit_store_memory_emitter::emit_data() const {
    // No additional data emission needed for basic store
}

/* ============== jit_load_broadcast_emitter =============== */

jit_load_broadcast_emitter::jit_load_broadcast_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                                                       ov::intel_cpu::riscv64::cpu_isa_t isa,
                                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::gpr_to_vec) {
    bool is_supported_precision =
        any_of(dst_prc, ov::element::f32, ov::element::i32) && src_prc == dst_prc;
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");

    const auto broadcast_load = ov::as_type_ptr<snippets::op::BroadcastLoad>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(broadcast_load != nullptr, "Expects BroadcastLoad expression");
    count = 1;  // BroadcastLoad loads a single scalar value
    byte_size = src_prc.size();
}

void jit_load_broadcast_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == ov::intel_cpu::riscv64::gv) {
        emit_isa<ov::intel_cpu::riscv64::gv>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_broadcast_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    auto src_gpr = Xbyak_riscv::Reg(in[0]);
    auto dst_vreg = Xbyak_riscv::VReg(out[0]);

    // Set vector configuration for appropriate element size
    if (byte_size == 4) {
        h->vsetivli(Xbyak_riscv::zero, 4, Xbyak_riscv::SEW::e32, Xbyak_riscv::LMUL::m1);
    } else if (byte_size == 2) {
        h->vsetivli(Xbyak_riscv::zero, 4, Xbyak_riscv::SEW::e16, Xbyak_riscv::LMUL::m1);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported byte size: ", byte_size);
    }

    // Load scalar from memory and broadcast to vector register
    // First load the scalar value into a temporary GPR
    auto tmp_gpr = Xbyak_riscv::Reg(aux_gpr_idxs.empty() ? Xbyak_riscv::t0.getIdx() : aux_gpr_idxs[0]);
    
    // Calculate effective address if there's an offset
    if (compiled_byte_offset == 0) {
        if (byte_size == 2) {
            h->lhu(tmp_gpr, src_gpr, 0);
        } else {
            h->lw(tmp_gpr, src_gpr, 0);
        }
    } else {
        auto addr_gpr = Xbyak_riscv::Reg(aux_gpr_idxs.size() > 1 ? aux_gpr_idxs[1] : Xbyak_riscv::t1.getIdx());
        h->addi(addr_gpr, src_gpr, static_cast<int32_t>(compiled_byte_offset));
        if (byte_size == 2) {
            h->lhu(tmp_gpr, addr_gpr, 0);
        } else {
            h->lw(tmp_gpr, addr_gpr, 0);
        }
    }
    
    // Move scalar to vector register and broadcast
    h->vmv_v_x(dst_vreg, tmp_gpr);  // Broadcast scalar to all elements
}

void jit_load_broadcast_emitter::emit_data() const {
    // No additional data emission needed for broadcast load operations
}

}  // namespace ov::intel_cpu::riscv64
