// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <memory>
#include <string>
#include <vector>
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <cstring>

#    include "emitters/plugin/aarch64/jit_emitter.hpp"
#    include "openvino/runtime/threading/thread_local.hpp"

namespace ov::intel_cpu::aarch64 {

using ov::threading::ThreadLocal;

class jit_uni_segfault_detector_emitter;
extern const std::shared_ptr<ThreadLocal<jit_uni_segfault_detector_emitter*>> g_custom_segfault_handler;

class jit_uni_segfault_detector_emitter : public jit_emitter {
public:
    jit_uni_segfault_detector_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                      jit_emitter* target_emitter,
                                      bool is_load,
                                      bool is_store,
                                      std::string target_node_name);

    size_t get_inputs_count() const override;

    const jit_emitter* get_target_emitter() const;

    std::string info() const;

private:
    // Emit code to save "this" pointer to global handler, then track memory address, etc.
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    jit_emitter* m_target_emitter = nullptr;
    bool is_target_use_load_emitter = false;
    bool is_target_use_store_emitter = false;
    std::string m_target_node_name;

    void save_target_emitter() const;
    static void set_local_handler(jit_uni_segfault_detector_emitter* emitter_address);
    void memory_track(size_t gpr_idx_for_mem_address) const;

    mutable size_t start_address = 0;
    mutable size_t current_address = 0;
    mutable size_t iteration = 0;

#    ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_uni_segfault_detector_emitter(const jit_uni_segfault_detector_emitter* emitter);
#    endif
};

}  // namespace ov::intel_cpu::aarch64

#endif
