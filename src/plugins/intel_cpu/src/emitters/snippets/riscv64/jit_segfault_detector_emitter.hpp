// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <string>
#include <vector>
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <cstring>

#    include "emitters/plugin/riscv64/jit_emitter.hpp"
#    include "emitters/snippets/common/jit_segfault_detector_emitter_base.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_uni_segfault_detector_emitter;

class jit_uni_segfault_detector_emitter
    : public ov::intel_cpu::jit_segfault_detector_emitter_base<jit_uni_segfault_detector_emitter, jit_emitter> {
public:
    using base_t = ov::intel_cpu::jit_segfault_detector_emitter_base<jit_uni_segfault_detector_emitter, jit_emitter>;

    jit_uni_segfault_detector_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                      ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                      jit_emitter* target_emitter,
                                      bool is_load,
                                      bool is_store,
                                      std::string target_node_name);

    size_t get_inputs_num() const override;
    size_t aux_gprs_count() const override;

    const jit_emitter* get_target_emitter() const;

private:
    void save_target_emitter() const override;
    static void set_local_handler(jit_uni_segfault_detector_emitter* emitter_address);
    void memory_track(size_t gpr_idx_for_mem_address) const override;

    friend std::string init_info_jit_uni_segfault_detector_emitter(const jit_uni_segfault_detector_emitter* emitter);
};

}  // namespace ov::intel_cpu::riscv64

#endif
