// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/plugin/aarch64/jit_load_store_emitters.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_memory_emitter : public jit_emitter  {
public:
    jit_memory_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                  dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    ov::element::Type src_prc;
    ov::element::Type dst_prc;

    size_t count = 0;
    size_t byte_offset = 0;
};

class jit_load_memory_emitter : public jit_memory_emitter {
public:
    jit_load_memory_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
};

class jit_load_broadcast_emitter : public jit_memory_emitter {
public:
    jit_load_broadcast_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                         dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class jit_store_memory_emitter : public jit_memory_emitter  {
public:
    jit_store_memory_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                 dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                 const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
