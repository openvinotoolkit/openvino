// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_fill_emitter : public jit_emitter {
public:
    jit_fill_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                     dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 1;}

protected:
    size_t get_aux_gprs_count() const override;

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void fill_full(const std::vector<size_t> &out) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void fill_tail(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

    bool is_full_reg() const { return offset == 0; }
    bool is_optimized() const { return is_full_reg() && fill_value == uint32_t(0x0); }

    size_t offset = 0;
    uint32_t fill_value = 0x0;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
