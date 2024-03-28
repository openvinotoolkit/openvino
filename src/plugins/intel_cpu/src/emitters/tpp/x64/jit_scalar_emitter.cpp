// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_scalar_emitter.hpp"
#include "emitters/snippets/x64/jit_snippets_emitters.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

ScalarTppEmitter::ScalarTppEmitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    push_arg_entry_of("scalar_tpp", jit_scalar_emitter::read_value(expr), false);
    in_out_type_ = gpr_to_gpr;
}

void ScalarTppEmitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    const auto it = entry_map_.find("scalar_tpp");
    OV_CPU_JIT_EMITTER_ASSERT(it != entry_map_.end(), "Value has not been found in the table");
    const auto& out_reg = Reg64(static_cast<int>(out[0]));
    h->mov(out_reg, p_table);
    h->add(out_reg, (*it).second.off);
}

}  // namespace intel_cpu
}  // namespace ov
