// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "snippets/lowered/expression.hpp"
#include "openvino/op/cyberspore_tssn.hpp"

namespace ov::intel_cpu {

class jit_nop_emitter : public jit_emitter {
public:
    jit_nop_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                    dnnl::impl::cpu::x64::cpu_isa_t isa,
                    const ov::snippets::lowered::ExpressionPtr& expr,
                    emitter_in_out_map emitter_type = gpr_to_gpr);

    size_t get_inputs_num() const override {
        return 0;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override {}
};

class jit_parameter_emitter : public jit_nop_emitter {
public:
    jit_parameter_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                          dnnl::impl::cpu::x64::cpu_isa_t isa,
                          const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }
};

class jit_result_emitter : public jit_nop_emitter {
public:
    jit_result_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                       dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override {
        return 1;
    }
};

class jit_broadcast_move_emitter : public jit_emitter {
public:
    jit_broadcast_move_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                               dnnl::impl::cpu::x64::cpu_isa_t isa,
                               const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    size_t byte_size = 0LU;
};

class jit_scalar_emitter : public jit_emitter {
public:
    jit_scalar_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                       dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }
    size_t aux_gprs_count() const override {
        return 1;
    }
    static int32_t read_value(const ov::snippets::lowered::ExpressionPtr& expr);

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const;
};

class jit_cyberspore_tssn_emitter : public jit_emitter {
public:
    jit_cyberspore_tssn_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                                dnnl::impl::cpu::x64::cpu_isa_t isa,
                                const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 3;
    }

    size_t aux_vecs_count() const override;

    static std::set<std::vector<ov::element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node);

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    void register_table_entries() override;

    float m_setpoint = 0.f;
    float m_decay_rate = 0.f;
    int32_t m_decay_gate = 0;
};

}  // namespace ov::intel_cpu
