// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_memory_emitters.hpp"

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"
#include "snippets/op/buffer.hpp"


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_memory_emitter::jit_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr, emitter_in_out_map in_out_type)
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
        OV_CPU_JIT_EMITTER_ASSERT(memory_access->is_memory_access_output_port(0), "must be output port - memory access");
        count = memory_access->get_output_count();
        compiled_byte_offset = memory_access->get_output_offset();
        buffer_cluster_id = get_consumer_buffer_cluster_id(expr);
    } else {
        OV_CPU_JIT_EMITTER_THROW("unsupported in_out_type");
    }

    if (ov::snippets::utils::is_dynamic_value(compiled_byte_offset)) {
        is_offset_runtime = true;
        // Compiled byte offset is zero to manually `add` runtime offset before operation and `sub` after to reset pointer in the register
        compiled_byte_offset = 0;
        OV_CPU_JIT_EMITTER_ASSERT(buffer_cluster_id != SIZE_MAX, "Incorrect buffer offset in call_args");
    }
}

size_t jit_memory_emitter::aux_gprs_count() const {
    // for runtime arguments
    return is_offset_runtime ? 1 : 0;
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
    for (const auto& consumer : consumers)
        if (const auto buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(consumer.get_expr()))
            return buffer->get_cluster_id();
    return SIZE_MAX;
}

std::vector<size_t> jit_memory_emitter::get_available_aux_gprs() const {
    OV_CPU_JIT_EMITTER_ASSERT(IMPLICATION(is_offset_runtime, !aux_gpr_idxs.empty()),
                              "If offset is dynamic, memory emitter need to have one aux gpr at least!");
    auto available_aux_gprs = aux_gpr_idxs;
    if (is_offset_runtime)
        available_aux_gprs.pop_back();
    return available_aux_gprs;
}

void jit_memory_emitter::emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);

    Reg64 reg_runtime_params = abi_param1;  // defined by jit_kernel_emitter
    Reg64 aux_gpr = is_offset_runtime ? Reg64(static_cast<int>(aux_gpr_idxs.back())) : Reg64();

    Reg64 data_reg;
    if (in_out_type_ == emitter_in_out_map::gpr_to_vec) {
        data_reg = Reg64(in_idxs[0]);
    } else if (in_out_type_ == emitter_in_out_map::vec_to_gpr) {
        data_reg = Reg64(out_idxs[0]);
    } else {
        OV_CPU_JIT_EMITTER_THROW("unsupported in_out_type");
    }

    if (is_offset_runtime) {
        h->mov(aux_gpr, h->ptr[reg_runtime_params + GET_OFF(buffer_offsets) + buffer_cluster_id * sizeof(size_t)]);
        h->add(data_reg, aux_gpr);
    }

    emit_impl(in_idxs, out_idxs);

    if (is_offset_runtime) {
        h->sub(data_reg, aux_gpr);
    }

    emitter_postamble();
}

jit_load_memory_emitter::jit_load_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::gpr_to_vec) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::Load>(expr->get_node()), "expects Load node");
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void jit_load_memory_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(load_emitter, "Load CPU emitter isn't initialized!");
    load_emitter->emit_code({in[0], compiled_byte_offset}, {out[0]}, aux_vec_idxs, get_available_aux_gprs());
}

void jit_load_memory_emitter::emit_data() const {
    load_emitter->emit_data();
}

jit_load_broadcast_emitter::jit_load_broadcast_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::gpr_to_vec) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::BroadcastLoad>(expr->get_node()), "expects BroadcastLoad node");
    if (src_prc != dst_prc)
        OV_CPU_JIT_EMITTER_THROW("supports only equal input and output types but gets: ",
                                 src_prc.get_type_name(),
                                 " and ",
                                 dst_prc.get_type_name());
}

void jit_load_broadcast_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
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
void jit_load_broadcast_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(in[0]);
    Vmm vmm_dst = Vmm(out[0]);

    // It doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
    // key point here is not to add post-increment, it might be fixed by some other approach in future
    switch (src_prc.size()) {
        case 4: h->uni_vbroadcastss(vmm_dst, h->ptr[in_reg + compiled_byte_offset]); break;
        case 2: h->vpbroadcastw(vmm_dst, h->ptr[in_reg + compiled_byte_offset]); break;
        case 1: h->vpbroadcastb(vmm_dst, h->ptr[in_reg + compiled_byte_offset]); break;
        default: OV_CPU_JIT_EMITTER_THROW("Unsupported data type");
    }
}

jit_store_memory_emitter::jit_store_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr, emitter_in_out_map::vec_to_gpr) {
    if (ov::is_type<ov::intel_cpu::StoreConvertTruncation>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::truncation));
    } else if (ov::is_type<ov::intel_cpu::StoreConvertSaturation>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::saturation));
    } else if (ov::is_type<ov::snippets::op::Store>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count));
    } else {
        OV_CPU_JIT_EMITTER_THROW("expects Store node");
    }
}

void jit_store_memory_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(store_emitter, "Store CPU emitter isn't initialized!");
    store_emitter->emit_code({in[0]}, {out[0], compiled_byte_offset}, aux_vec_idxs, get_available_aux_gprs());
}

void jit_store_memory_emitter::emit_data() const {
    store_emitter->emit_data();
}

}   // namespace intel_cpu
}   // namespace ov
