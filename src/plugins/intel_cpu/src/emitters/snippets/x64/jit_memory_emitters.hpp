// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"


namespace ov {
namespace intel_cpu {

class jit_memory_emitter : public jit_emitter  {
public:
    jit_memory_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr, emitter_in_out_map in_out_type);

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

protected:
    static size_t get_parent_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr);
    static size_t get_consumer_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr);

    size_t aux_gprs_count() const override;

    std::vector<size_t> get_available_aux_gprs() const;

    ov::element::Type src_prc;
    ov::element::Type dst_prc;

    size_t count = 0;
    size_t compiled_byte_offset = 0;
    size_t buffer_cluster_id = 0;
    bool is_offset_runtime = false;

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_memory_emitter(const jit_memory_emitter *emitter);
#endif
};

class jit_load_memory_emitter : public jit_memory_emitter {
public:
    jit_load_memory_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

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

class jit_store_memory_emitter : public jit_memory_emitter  {
public:
    jit_store_memory_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void emit_data() const override;

private:
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};

}   // namespace intel_cpu
}   // namespace ov
