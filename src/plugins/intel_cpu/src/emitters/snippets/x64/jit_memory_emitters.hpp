// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/common/x64/jit_emitter.hpp"
#include "emitters/common/x64/jit_load_store_emitters.hpp"


namespace ov {
namespace intel_cpu {

///
/// Memory emitters:
///
/// *Note*: post increment is embedded into Load/Store operation which means that
/// it's illigal to load/store to the same address multiple times
/// Typical application can be if Load and BroadcastLoad are performed from the same pointer.
/// If Load goes before BroadcastLoad topologicaly the resilt will be incorrect
/// For scalar loads we can use different loops. Tiling indeed can be arbitrary and post increment should be somehow coded into ISA.
/// Blocked parameter to tell if input is actually blocked. Broadcast means broadcast by W in other cases no need to substitute load.
class jit_memory_emitter : public jit_emitter  {
public:
    jit_memory_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    ov::element::Type src_prc;
    ov::element::Type dst_prc;

    size_t count = 0;
    size_t byte_offset = 0;
};

class jit_load_memory_emitter : public jit_memory_emitter {
public:
    jit_load_memory_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
};

class jit_load_broadcast_emitter : public jit_memory_emitter {
public:
    jit_load_broadcast_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                               const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class jit_load_convert_emitter : public jit_memory_emitter {
public:
    jit_load_convert_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
};

class jit_store_memory_emitter : public jit_memory_emitter  {
public:
    jit_store_memory_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};

class jit_store_convert_emitter : public jit_memory_emitter {
public:
    jit_store_convert_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};

}   // namespace intel_cpu
}   // namespace ov
