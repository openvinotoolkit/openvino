// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef SNIPPETS_DEBUG_CAPS

#    include <cstddef>
#    include <memory>
#    include <optional>
#    include <string>
#    include <vector>

#    include "openvino/runtime/threading/thread_local.hpp"

namespace ov::intel_cpu {

template <class Derived>
inline const std::shared_ptr<ov::threading::ThreadLocal<Derived*>> g_custom_segfault_handler =
    std::make_shared<ov::threading::ThreadLocal<Derived*>>();

// Shared CRTP-style base to de-duplicate the common segfault detector bookkeeping
// across architectures. Architecture-specific emitters still provide:
// 1) save_target_emitter()
// 2) memory_track(gpr_idx_for_mem_address)
// 3) (optionally) save_before_memory_track() policy override
template <class Derived, class JitEmitterT>
class jit_segfault_detector_emitter_base : public JitEmitterT {
public:
    using handler_t = ov::threading::ThreadLocal<Derived*>;

    template <class HostT, class IsaT>
    jit_segfault_detector_emitter_base(HostT* host,
                                       IsaT host_isa,
                                       JitEmitterT* target_emitter,
                                       bool is_load,
                                       bool is_store,
                                       std::string target_node_name)
        : JitEmitterT(host, host_isa),
          m_target_emitter(target_emitter),
          is_target_use_load_emitter(is_load),
          is_target_use_store_emitter(is_store),
          m_target_node_name(std::move(target_node_name)) {}

    const JitEmitterT* get_target_emitter() const {
        return m_target_emitter;
    }

protected:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override {
        const auto mem_gpr_idx = resolve_tracked_gpr_idx(in_vec_idxs, out_vec_idxs);
        if (!mem_gpr_idx.has_value()) {
            save_target_emitter();
            return;
        }

        if (save_before_memory_track()) {
            save_target_emitter();
        }
        memory_track(*mem_gpr_idx);
    }

    // Default policy mirrors the existing x64/riscv64 behavior.
    virtual bool save_before_memory_track() const {
        return true;
    }

    static void set_local_handler_impl(const std::shared_ptr<handler_t>& handler, Derived* emitter_address) {
        handler->local() = emitter_address;
    }

    JitEmitterT* m_target_emitter = nullptr;
    bool is_target_use_load_emitter = false;
    bool is_target_use_store_emitter = false;
    std::string m_target_node_name;

    mutable size_t start_address = 0;
    mutable size_t current_address = 0;
    mutable size_t iteration = 0;

private:
    std::optional<size_t> resolve_tracked_gpr_idx(const std::vector<size_t>& in_vec_idxs,
                                                  const std::vector<size_t>& out_vec_idxs) const {
        if (is_target_use_load_emitter && !in_vec_idxs.empty()) {
            return in_vec_idxs[0];
        }
        if (is_target_use_store_emitter && !out_vec_idxs.empty()) {
            return out_vec_idxs[0];
        }
        return std::nullopt;
    }

    virtual void save_target_emitter() const = 0;
    virtual void memory_track(size_t gpr_idx_for_mem_address) const = 0;
};

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
