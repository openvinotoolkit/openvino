// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef SNIPPETS_DEBUG_CAPS

#    include <cstddef>
#    include <cstdint>
#    include <memory>
#    include <utility>
#    include <vector>

namespace ov::intel_cpu {

template <typename JitEmitterT>
class jit_debug_emitter_base_common : public JitEmitterT {
public:
    enum class EmissionLocation : uint8_t { preamble, postamble, both };

    template <typename HostT, typename IsaT, typename ExecPrcT, typename InOutMapT>
    jit_debug_emitter_base_common(HostT* host,
                                  IsaT host_isa,
                                  ExecPrcT exec_prc,
                                  InOutMapT in_out_type,
                                  std::shared_ptr<JitEmitterT> target_emitter,
                                  std::shared_ptr<JitEmitterT> decorator_emitter,
                                  EmissionLocation loc)
        : JitEmitterT(host, host_isa, exec_prc, in_out_type),
          m_target_emitter(std::move(target_emitter)),
          m_decorator_emitter(std::move(decorator_emitter)),
          m_decorator_emit_loc(loc) {
        this->prepare_table();
    }

    void emit_data() const override {
        m_target_emitter->emit_data();
    }

protected:
    void prepare_table() override {
        m_target_emitter->prepare_table();
    }

    void register_table_entries() override {
        m_target_emitter->register_table_entries();
    }

    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override {
        m_target_emitter->emit_impl(in_idxs, out_idxs);
    }

    void emitter_postamble() const override {
        m_target_emitter->emitter_postamble();
    }

    void validate_arguments(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override {
        m_target_emitter->validate_arguments(in_idxs, out_idxs);
    }

    template <typename... EmitArgs>
    void emit_code_with_decorator(const EmitArgs&... args) const {
        if (emit_before_target()) {
            m_decorator_emitter->emit_code(args...);
        }

        m_target_emitter->emit_code(args...);

        if (emit_after_target()) {
            m_decorator_emitter->emit_code(args...);
        }
    }

    const std::shared_ptr<JitEmitterT> m_target_emitter;

private:
    [[nodiscard]] bool emit_before_target() const {
        return m_decorator_emit_loc == EmissionLocation::preamble || m_decorator_emit_loc == EmissionLocation::both;
    }

    [[nodiscard]] bool emit_after_target() const {
        return m_decorator_emit_loc == EmissionLocation::postamble || m_decorator_emit_loc == EmissionLocation::both;
    }

    const std::shared_ptr<JitEmitterT> m_decorator_emitter;
    EmissionLocation m_decorator_emit_loc;
};

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
