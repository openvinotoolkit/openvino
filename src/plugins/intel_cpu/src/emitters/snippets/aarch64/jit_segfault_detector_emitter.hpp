// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <memory>
#include <string>
#include <vector>

#ifdef SNIPPETS_DEBUG_CAPS

#    include <cstring>

#    include "emitters/plugin/aarch64/jit_emitter.hpp"
#    include "openvino/runtime/threading/thread_local.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace ov::threading;

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

private:
    // emit code is to save "this" pointer(jit_uni_segfault_detector_emitter) to global handler, then print info w/ it's
    // target_emitter. and to save tracked memory address, iteration, etc to print
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

    friend std::string init_info_jit_uni_segfault_detector_emitter(const jit_uni_segfault_detector_emitter* emitter);
};

// Returns a snapshot string with start/current addresses and iteration
std::string get_segfault_tracking_info();

}  // namespace ov::intel_cpu::aarch64

#endif
