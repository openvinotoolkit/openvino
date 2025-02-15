// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <utility>

#    include "emitters/plugin/x64/jit_emitter.hpp"

namespace ov::intel_cpu {

class jit_debug_emitter : public jit_emitter {
public:
    enum class EmissionLocation { preamble, postamble, both };
    jit_debug_emitter(const std::shared_ptr<jit_emitter>& target_emitter,
                      std::shared_ptr<jit_emitter> decorator_emitter,
                      const EmissionLocation& loc)
        : jit_emitter(target_emitter->h,
                      target_emitter->host_isa_,
                      target_emitter->exec_prc_,
                      target_emitter->in_out_type_),
          m_target_emitter(target_emitter),
          m_decorator_emitter(std::move(decorator_emitter)),
          m_decorator_emit_loc(loc) {
        prepare_table();
    }

    void emit_data() const override;

    size_t get_inputs_num() const override;
    size_t aux_vecs_count() const override;

protected:
    size_t aux_gprs_count() const override;

    void prepare_table() override;
    void register_table_entries() override;

    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    void emitter_preamble(const std::vector<size_t>& in_idxs,
                          const std::vector<size_t>& out_idxs,
                          const std::vector<size_t>& pool_vec_idxs,
                          const std::vector<size_t>& pool_gpr_idxs) const override;
    void emitter_postamble() const override;

private:
    void validate_arguments(const std::vector<size_t>& arg0, const std::vector<size_t>& arg1) const override;
    // wrapper emitter for product function
    const std::shared_ptr<jit_emitter> m_target_emitter;
    // debug capability emitter
    const std::shared_ptr<jit_emitter> m_decorator_emitter;

    EmissionLocation m_decorator_emit_loc;
};

}  // namespace ov::intel_cpu

#endif