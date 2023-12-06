// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_memory_emitters.hpp"

#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_memory_emitter::jit_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    src_prc = n->get_input_element_type(0);
    dst_prc = n->get_output_element_type(0);
}

jit_load_memory_emitter::jit_load_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("jit_load_memory_emitter supports only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    const auto load = std::dynamic_pointer_cast<snippets::op::Load>(expr->get_node());
    count = load->get_count();
    byte_offset = load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void jit_load_memory_emitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("Load emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_memory_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        OPENVINO_THROW("Load CPU emitter isn't initialized for jit_load_memory_emitter!");
    load_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void jit_load_memory_emitter::emit_data() const {
    load_emitter->emit_data();
}

jit_load_broadcast_emitter::jit_load_broadcast_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("BroadcastEmitters support only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    const auto broadcast_load = std::dynamic_pointer_cast<snippets::op::BroadcastLoad>(expr->get_node());
    byte_offset = broadcast_load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
}

void jit_load_broadcast_emitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("BroadcastLoad emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_broadcast_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(in[0]);
    Vmm vmm_dst = Vmm(out[0]);

    // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
    // key point here is not to add post-increment, it might be fixed by some other approach in future
    switch (src_prc.size()) {
        case 4: h->uni_vbroadcastss(vmm_dst, h->ptr[in_reg + byte_offset]); break;
        case 2: h->vpbroadcastw(vmm_dst, h->ptr[in_reg + byte_offset]); break;
        case 1: h->vpbroadcastb(vmm_dst, h->ptr[in_reg + byte_offset]); break;
        default: assert(!"unsupported data type");
    }
}

jit_load_convert_emitter::jit_load_convert_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr) {
    const auto load = ov::as_type_ptr<snippets::op::Load>(expr->get_node());
    count = load->get_count();
    byte_offset = load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void jit_load_convert_emitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("LoadConvert emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_convert_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        OPENVINO_THROW("Load CPU emitter isn't initialized for jit_load_memory_emitter!");
    load_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void jit_load_convert_emitter::emit_data() const {
    load_emitter->emit_data();
}

jit_store_memory_emitter::jit_store_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_memory_emitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("jit_store_memory_emitter supports only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    const auto store = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    count = store->get_count();
    byte_offset = store->get_offset();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
    store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count));
}

void jit_store_memory_emitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("Store emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_store_memory_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        OPENVINO_THROW("Store CPU emitter isn't initialized for jit_store_memory_emitter!");
    store_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void jit_store_memory_emitter::emit_data() const {
    store_emitter->emit_data();
}

jit_store_convert_emitter::jit_store_convert_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_memory_emitter(h, isa, expr) {
    const auto store = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    count = store->get_count();
    byte_offset = store->get_offset();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;

    if (ov::is_type<ov::intel_cpu::StoreConvertTruncation>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::truncation));
    } else if (ov::is_type<ov::intel_cpu::StoreConvertSaturation>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::saturation));
    }
}

void jit_store_convert_emitter::emit_impl(const std::vector<size_t>& in,
                                    const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("StoreConvert emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_store_convert_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        OPENVINO_THROW("Store CPU emitter isn't initialized for jit_store_memory_emitter!");
    store_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void jit_store_convert_emitter::emit_data() const {
    store_emitter->emit_data();
}

}   // namespace intel_cpu
}   // namespace ov
