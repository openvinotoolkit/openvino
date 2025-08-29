// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "jit_loop_base_emitters.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu {

class jit_loop_begin_emitter : public jit_loop_begin_base_emitter {
public:
    jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                           dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    size_t m_work_amount = 0;
};

class jit_loop_end_emitter : public jit_loop_end_base_emitter {
public:
    jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                         dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
};

}  // namespace ov::intel_cpu
