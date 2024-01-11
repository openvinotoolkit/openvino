// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

namespace ov {
namespace intel_cpu {

enum emit_debug_loc {
    pre_emit,
    post_emit,
    both_emit,
};

class jit_debug_emitter : public jit_emitter {
public:
    jit_debug_emitter(const std::shared_ptr<jit_emitter>& target_emitter, const std::shared_ptr<jit_emitter>& decorator_emitter, const emit_debug_loc& loc)
        : jit_emitter(target_emitter->h, target_emitter->host_isa_, target_emitter->exec_prc_, target_emitter->in_out_type_),
        m_target_emitter(target_emitter), m_decorator_emitter(decorator_emitter), m_emit_debug_loc(loc) {
            prepare_table();
        }

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;
    void emit_data() const override;

    size_t get_inputs_num() const override;
    size_t aux_vecs_count() const override;

protected:
    size_t aux_gprs_count() const override;

    void prepare_table() override;
    void register_table_entries() override;

    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;

    void emitter_preamble(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                          const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const override;
    void emitter_postamble() const override;

private:
    void validate_arguments(const std::vector<size_t>& arg0, const std::vector<size_t>& arg1) const override;

    const std::shared_ptr<jit_emitter> m_target_emitter;
    const std::shared_ptr<jit_emitter> m_decorator_emitter;

    emit_debug_loc m_emit_debug_loc;
};

}   // namespace intel_cpu
}   // namespace ov

#endif