// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h"

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

NopEmitter::NopEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : aarch64::jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

BroadcastMoveEmitter::BroadcastMoveEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    if (n->get_input_element_type(0) != n->get_output_element_type(0))
        OPENVINO_THROW("BroadcastMoveEmitter supports only equal input and output types but gets: ",
                       n->get_input_element_type(0),
                       " and ",
                       n->get_output_element_type(0));
    if (n->get_input_element_type(0) != ov::element::f32)
        OPENVINO_THROW("BroadcastMoveEmitter only supports FP32 precision.");

    byte_size = n->get_input_element_type(0).size();
}

void BroadcastMoveEmitter::emit_impl(const std::vector<size_t>& in,
          const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("BroadcastMove emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void BroadcastMoveEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in[0]);
    TReg dst = TReg(out[0]);

    switch (byte_size) {
        case 4:
            h->dup(dst.s, src.s[0]);
            break;
        default:
            OPENVINO_THROW("unsupported data size ", byte_size);
    }
}

ScalarEmitter::ScalarEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    const auto& precision = n->get_output_element_type(0);
    switch (precision) {
        case element::i32: {
            value = ov::as_type_ptr<ov::op::v0::Constant>(n)->cast_vector<int32_t>()[0];
            break;
        }
        case element::f32: {
            value = dnnl::impl::float2int(ov::as_type_ptr<ov::op::v0::Constant>(n)->cast_vector<float>()[0]);
            break;
        }
        default: {
            OPENVINO_THROW("Scalar emitter doesn't support ", precision);
        }
    }
    push_arg_entry_of("scalar", value, true);
    prepare_table();
}

void ScalarEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("Scalar emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void ScalarEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg dst = TReg(out[0]);
    AdrImm src = table_val("scalar");

    h->uni_ld1rw(dst.s, src.getXn(), src.getImm());
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
