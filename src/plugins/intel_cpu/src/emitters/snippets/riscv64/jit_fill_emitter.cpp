// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_fill_emitter.hpp"

#include <cstddef>
#include <cstdint>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/fill.hpp"
#include "utils.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_csr.hpp"

namespace ov::intel_cpu::riscv64 {

using jit_generator_t = ov::intel_cpu::riscv64::jit_generator_t;
using cpu_isa_t = ov::intel_cpu::riscv64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_fill_emitter::jit_fill_emitter(jit_generator_t* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa, ov::element::f32, emitter_in_out_map::vec_to_vec) {
    const auto fill = ov::as_type_ptr<snippets::op::Fill>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(fill != nullptr, "Expects Fill expression");
    OV_CPU_JIT_EMITTER_ASSERT(fill->get_element_type().size() == 4,
                              "Supports only 4 Byte element types but gets ",
                              fill->get_element_type());

    offset = fill->get_offset();
    fill_value = fill->get_fill_value();
}

size_t jit_fill_emitter::aux_gprs_count() const {
    return is_optimized() && utils::get_snippet_lanes() <= 31 ? 0 : 1;
}

void jit_fill_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == ov::intel_cpu::riscv64::gv) {
        emit_isa<ov::intel_cpu::riscv64::gv>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Fill emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_fill_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    const auto supported_et_count = utils::get_snippet_lanes();
    OPENVINO_ASSERT(offset <= supported_et_count,
                    "Fill emitter offset ",
                    offset,
                    " exceeds register capacity ",
                    supported_et_count);

    set_vector_length(h, supported_et_count, Xbyak_riscv::SEW::e32, aux_gpr_idxs);

    if (is_full_reg()) {
        fill_full<isa>(out);
    } else {
        fill_tail<isa>(in, out);
    }
}

template <cpu_isa_t isa>
void jit_fill_emitter::fill_full(const std::vector<size_t>& out) const {
    auto dst = Xbyak_riscv::VReg(out[0]);

    if (is_optimized()) {
        h->vmv_v_x(dst, Xbyak_riscv::zero);
        return;
    }

    OPENVINO_ASSERT(!aux_gpr_idxs.empty(), "Fill emitter expects one auxiliary GPR register");
    const auto fill_reg = Xbyak_riscv::Reg(static_cast<int>(aux_gpr_idxs[0]));
    h->uni_li(fill_reg, static_cast<int64_t>(fill_value));
    h->vmv_v_x(dst, fill_reg);
}

template <cpu_isa_t isa>
void jit_fill_emitter::fill_tail(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    const auto supported_et_count = utils::get_snippet_lanes();
    const auto stack_size = static_cast<int>(supported_et_count * sizeof(uint32_t));

    auto src = Xbyak_riscv::VReg(in[0]);
    auto dst = Xbyak_riscv::VReg(out[0]);
    if (src.getIdx() != dst.getIdx()) {
        h->vmv_v_v(dst, src);
    }

    if (offset == supported_et_count) {
        return;
    }

    OPENVINO_ASSERT(!aux_gpr_idxs.empty(), "Fill emitter expects one auxiliary GPR register");
    const auto fill_reg = Xbyak_riscv::Reg(static_cast<int>(aux_gpr_idxs[0]));
    h->uni_li(fill_reg, static_cast<int64_t>(fill_value));

    // Write vector to stack, patch tail elements and load vector back.
    h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -stack_size);
    h->vse32_v(dst, Xbyak_riscv::sp);
    for (size_t i = offset; i < supported_et_count; i++) {
        h->sw(fill_reg, Xbyak_riscv::sp, static_cast<int32_t>(i * sizeof(uint32_t)));
    }
    h->vle32_v(dst, Xbyak_riscv::sp);
    h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, stack_size);
}

}  // namespace ov::intel_cpu::riscv64
