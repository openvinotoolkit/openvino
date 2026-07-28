// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <utility>

#    include "emitters/plugin/aarch64/jit_emitter.hpp"
#    include "emitters/snippets/common/jit_debug_emitter_base.hpp"

namespace ov::intel_cpu::aarch64 {

template <typename JitEmitterT>
class jit_debug_emitter_aarch64_base : public ov::intel_cpu::jit_debug_emitter_base_common<JitEmitterT> {
public:
    using base_t = ov::intel_cpu::jit_debug_emitter_base_common<JitEmitterT>;
    using EmissionLocation = typename base_t::EmissionLocation;
    using base_t::base_t;

    [[nodiscard]] size_t get_inputs_count() const override {
        return this->m_target_emitter->get_inputs_count();
    }

    [[nodiscard]] size_t get_aux_vecs_count() const override {
        return this->m_target_emitter->get_aux_vecs_count();
    }

protected:
    [[nodiscard]] size_t get_aux_gprs_count() const override {
        return this->m_target_emitter->get_aux_gprs_count();
    }

    void emitter_preamble(const std::vector<size_t>& in_idxs,
                          const std::vector<size_t>& out_idxs,
                          const std::vector<size_t>& pool_vec_idxs,
                          const std::vector<size_t>& pool_gpr_idxs) const override {
        this->m_target_emitter->emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);
    }

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override {
        this->emit_code_with_decorator(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);
    }
};

class jit_debug_emitter : public jit_debug_emitter_aarch64_base<jit_emitter> {
public:
    using base_t = jit_debug_emitter_aarch64_base<jit_emitter>;
    using EmissionLocation = typename base_t::EmissionLocation;

    jit_debug_emitter(const std::shared_ptr<jit_emitter>& target_emitter,
                      std::shared_ptr<jit_emitter> decorator_emitter,
                      const EmissionLocation& loc)
        : base_t(target_emitter->h,
                 target_emitter->host_isa_,
                 target_emitter->exec_prc_,
                 target_emitter->in_out_type_,
                 target_emitter,
                 std::move(decorator_emitter),
                 loc) {}
};

}  // namespace ov::intel_cpu::aarch64

#endif
