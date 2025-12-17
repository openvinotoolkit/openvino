// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"

#include <xbyak/xbyak.h>

#include <climits>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cyberspore_tssn.hpp"
#include "snippets/lowered/expression.hpp"
#include "utils/ternary.hpp"

using namespace dnnl::impl::utils;

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

using jit_generator_t = dnnl::impl::cpu::x64::jit_generator_t;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_nop_emitter::jit_nop_emitter(jit_generator_t* h,
                                 cpu_isa_t isa,
                                 [[maybe_unused]] const ExpressionPtr& expr,
                                 emitter_in_out_map emitter_type)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_type;
}

jit_parameter_emitter::jit_parameter_emitter(jit_generator_t* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_nop_emitter(h, isa, expr) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

jit_result_emitter::jit_result_emitter(jit_generator_t* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_nop_emitter(h, isa, expr) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

jit_broadcast_move_emitter::jit_broadcast_move_emitter(jit_generator_t* h, cpu_isa_t isa, const ExpressionPtr& expr)
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

jit_scalar_emitter::jit_scalar_emitter(jit_generator_t* h, cpu_isa_t isa, const ExpressionPtr& expr)
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
void jit_scalar_emitter::emit_isa(const std::vector<size_t>& /*in*/, const std::vector<size_t>& out) const {
    using Vmm = typename dnnl::impl::utils::
        conditional3<isa == dnnl::impl::cpu::x64::sse41, Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    auto vmm_dst = Vmm(out[0]);
    h->uni_vbroadcastss(vmm_dst, table_val("scalar"));
}

namespace {
inline size_t get_chunk_count(dnnl::impl::cpu::x64::cpu_isa_t isa) {
    switch (isa) {
    case dnnl::impl::cpu::x64::sse41:
        return 1;
    case dnnl::impl::cpu::x64::avx2:
        return 2;
    case dnnl::impl::cpu::x64::avx512_core:
        return 4;
    default:
        return 1;
    }
}

inline int get_cmp_predicate(bool is_positive) {
    return is_positive ? dnnl::impl::cpu::x64::jit_generator_t::_cmp_nle_us
                       : dnnl::impl::cpu::x64::jit_generator_t::_cmp_lt_os;
}
}  // namespace

jit_cyberspore_tssn_emitter::jit_cyberspore_tssn_emitter(jit_generator_t* h,
                                                         cpu_isa_t isa,
                                                         const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto node = ov::as_type_ptr<ov::op::v0::CybersporeTSSN>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(node, "jit_cyberspore_tssn_emitter expects CybersporeTSSN node");
    m_setpoint = node->get_homeostatic_setpoint();
    m_decay_rate = node->get_decay_rate();
    m_decay_gate = static_cast<int32_t>(ov::intel_cpu::ternary::decay_gate(m_decay_rate));
    prepare_table();
}

size_t jit_cyberspore_tssn_emitter::aux_vecs_count() const {
    return 11;  // constants + temporaries
}

std::set<std::vector<ov::element::Type>> jit_cyberspore_tssn_emitter::get_supported_precisions(
    const std::shared_ptr<ov::Node>& node) {
    OV_CPU_JIT_EMITTER_ASSERT(node, "Node is null");
    const auto cyberspore = ov::as_type_ptr<ov::op::v0::CybersporeTSSN>(node);
    OV_CPU_JIT_EMITTER_ASSERT(cyberspore, "Cyberspore emitter expects CybersporeTSSN node");

    const auto selective_type = cyberspore->get_input_element_type(2);
    OV_CPU_JIT_EMITTER_ASSERT(selective_type.is_real(),
                              "Selective input must be real, got ",
                              selective_type);

    return {{ov::element::t2, ov::element::t2, selective_type}};
}

void jit_cyberspore_tssn_emitter::register_table_entries() {
    push_arg_entry_of("setpoint", dnnl::impl::float2int(m_setpoint), true);
    push_arg_entry_of("pos_threshold", dnnl::impl::float2int(0.25f), true);
    push_arg_entry_of("neg_threshold", dnnl::impl::float2int(-0.25f), true);
    push_arg_entry_of("plus_one", 0x00000001, true);
    push_arg_entry_of("minus_one", 0xFFFFFFFF, true);
    push_arg_entry_of("zero_int", 0x00000000, true);
}

void jit_cyberspore_tssn_emitter::emit_impl(const std::vector<size_t>& in,
                                            const std::vector<size_t>& out) const {
    if (host_isa_ == cpu_isa_t::sse41) {
        emit_isa<cpu_isa_t::sse41>(in, out);
    } else if (host_isa_ == cpu_isa_t::avx2) {
        emit_isa<cpu_isa_t::avx2>(in, out);
    } else if (host_isa_ == cpu_isa_t::avx512_core) {
        emit_isa<cpu_isa_t::avx512_core>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA for Cyberspore snippets emitter");
    }
}

template <cpu_isa_t isa>
void jit_cyberspore_tssn_emitter::emit_isa(const std::vector<size_t>& in,
                                           const std::vector<size_t>& out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41,
                                                         Xmm,
                                                         isa == cpu_isa_t::avx2,
                                                         Ymm,
                                                         Zmm>::type;

    const auto events = Vmm(in[0]);
    const auto states = Vmm(in[1]);
    const auto selective = Vmm(in[2]);
    const auto dst = Vmm(out[0]);

    const auto setpoint = Vmm(aux_vec_idxs[0]);
    const auto pos_thr = Vmm(aux_vec_idxs[1]);
    const auto neg_thr = Vmm(aux_vec_idxs[2]);
    const auto plus_one = Vmm(aux_vec_idxs[3]);
    const auto minus_one = Vmm(aux_vec_idxs[4]);
    const auto zero_int = Vmm(aux_vec_idxs[5]);
    const auto decay_gate_vec = Vmm(aux_vec_idxs[6]);

    const auto tmp0 = Vmm(aux_vec_idxs[7]);
    const auto tmp1 = Vmm(aux_vec_idxs[8]);
    const auto tmp2 = Vmm(aux_vec_idxs[9]);
    const auto tmp3 = Vmm(aux_vec_idxs[10]);

    load_table_addr();
    h->uni_vbroadcastss(setpoint, table_val("setpoint"));
    h->uni_vbroadcastss(pos_thr, table_val("pos_threshold"));
    h->uni_vbroadcastss(neg_thr, table_val("neg_threshold"));
    h->uni_vpbroadcastd(plus_one, table_val("plus_one"));
    h->uni_vpbroadcastd(minus_one, table_val("minus_one"));
    h->uni_vpxor(zero_int, zero_int, zero_int);

    if (m_decay_gate == 0) {
        h->uni_vpxor(decay_gate_vec, decay_gate_vec, decay_gate_vec);
    } else if (m_decay_gate > 0) {
        h->uni_vmovups(decay_gate_vec, plus_one);
    } else {
        h->uni_vmovups(decay_gate_vec, minus_one);
    }

    h->uni_vpxor(dst, dst, dst);

    const auto chunk_count = get_chunk_count(host_isa_);

    auto extract_chunk = [&](const Vmm& src, const Xmm& dst_chunk, int chunk, bool fp) {
        if constexpr (isa == cpu_isa_t::sse41) {
            if (fp) {
                h->movaps(dst_chunk, Xmm(src.getIdx()));
            } else {
                h->movdqa(dst_chunk, Xmm(src.getIdx()));
            }
        } else if constexpr (isa == cpu_isa_t::avx2) {
            if (chunk == 0) {
                if (fp) {
                    h->vmovaps(dst_chunk, Xmm(src.getIdx()));
                } else {
                    h->vmovdqa(dst_chunk, Xmm(src.getIdx()));
                }
            } else {
                if (fp) {
                    h->vextractf128(dst_chunk, src, chunk);
                } else {
                    h->vextracti128(dst_chunk, src, chunk);
                }
            }
        } else {
            if (fp) {
                h->vextractf32x4(dst_chunk, src, chunk);
            } else {
                h->vextracti32x4(dst_chunk, src, chunk);
            }
        }
    };

    auto insert_chunk = [&](const Vmm& dst_vmm, const Xmm& chunk, int chunk_idx) {
        if constexpr (isa == cpu_isa_t::sse41) {
            h->movdqa(Xmm(dst_vmm.getIdx()), chunk);
        } else if constexpr (isa == cpu_isa_t::avx2) {
            h->vinserti128(dst_vmm, dst_vmm, chunk, chunk_idx);
        } else {
            h->vinserti32x4(dst_vmm, dst_vmm, chunk, chunk_idx);
        }
    };

    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const int chunk_id = static_cast<int>(chunk);

        auto selective_chunk = Xmm(tmp0.getIdx());
        extract_chunk(selective, selective_chunk, chunk_id, true);
        h->subps(selective_chunk, Xmm(setpoint.getIdx()));

        auto pos_mask = Xmm(tmp1.getIdx());
        h->movaps(pos_mask, selective_chunk);
        h->cmpps(pos_mask, Xmm(pos_thr.getIdx()), get_cmp_predicate(true));
        h->psrld(pos_mask, 31);

        auto neg_mask = Xmm(tmp2.getIdx());
        h->movaps(neg_mask, selective_chunk);
        h->cmpps(neg_mask, Xmm(neg_thr.getIdx()), get_cmp_predicate(false));
        h->psrad(neg_mask, 31);

        auto selective_int = Xmm(tmp0.getIdx());
        h->movdqa(selective_int, pos_mask);
        h->paddd(selective_int, neg_mask);

        auto events_chunk = Xmm(tmp1.getIdx());
        extract_chunk(events, events_chunk, chunk_id, false);
        auto event_int = Xmm(tmp1.getIdx());
        h->pmovsxbd(event_int, events_chunk);

        auto states_chunk = Xmm(tmp2.getIdx());
        extract_chunk(states, states_chunk, chunk_id, false);
        auto state_int = Xmm(tmp2.getIdx());
        h->pmovsxbd(state_int, states_chunk);

        auto result_int = Xmm(tmp3.getIdx());
        h->movdqa(result_int, event_int);
        h->pmulld(result_int, selective_int);

        auto decay_term = Xmm(tmp0.getIdx());
        h->movdqa(decay_term, state_int);
        h->pmulld(decay_term, Xmm(decay_gate_vec.getIdx()));

        h->paddd(result_int, decay_term);
        h->pmaxsd(result_int, Xmm(minus_one.getIdx()));
        h->pminsd(result_int, Xmm(plus_one.getIdx()));

        h->packssdw(result_int, Xmm(zero_int.getIdx()));
        h->packsswb(result_int, Xmm(zero_int.getIdx()));

        insert_chunk(dst, result_int, chunk_id);
    }
}

}  // namespace ov::intel_cpu
