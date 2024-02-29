// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class NopEmitter : public jit_emitter {
public:
    NopEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
               dnnl::impl::cpu::aarch64::cpu_isa_t isa,
               const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override {}
};

class BroadcastMoveEmitter : public jit_emitter {
public:
    BroadcastMoveEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                         dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

private:
    size_t byte_size = 0lu;
};

class ScalarEmitter : public jit_emitter {
public:
    ScalarEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                  dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 0;}

protected:
    size_t get_aux_gprs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

private:
    int32_t value;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
