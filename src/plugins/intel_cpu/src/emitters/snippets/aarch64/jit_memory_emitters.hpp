// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/plugin/aarch64/jit_load_store_emitters.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_memory_emitter : public jit_emitter {
public:
    jit_memory_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                       dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr,
                       emitter_in_out_map in_out_type);

    size_t get_aux_gprs_count() const override;

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    std::vector<size_t> get_available_aux_gprs() const;

protected:
    ov::element::Type src_prc;
    ov::element::Type dst_prc;

    size_t count = 0;
    size_t compiled_byte_offset = 0;
    size_t buffer_cluster_id = 0;
    bool is_offset_runtime = false;
};

class jit_load_memory_emitter : public jit_memory_emitter {
public:
    jit_load_memory_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                            dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_data() const override;

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
};

class jit_load_broadcast_emitter : public jit_memory_emitter {
public:
    jit_load_broadcast_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                               dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                               const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    size_t byte_size;
};

class jit_store_memory_emitter : public jit_memory_emitter {
public:
    jit_store_memory_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                             dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                             const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_data() const override;

    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};

}  // namespace ov::intel_cpu::aarch64
