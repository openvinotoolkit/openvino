// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_nop_emitter : public jit_emitter {
public:
    jit_nop_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                    dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                    const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 0;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override {}
};

class jit_broadcast_move_emitter : public jit_emitter {
public:
    jit_broadcast_move_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                               dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                               const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    size_t byte_size = 0lu;
};

class jit_scalar_emitter : public jit_emitter {
public:
    jit_scalar_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                       dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 0;
    }

protected:
    size_t get_aux_gprs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    int32_t value;
};

}  // namespace ov::intel_cpu::aarch64
