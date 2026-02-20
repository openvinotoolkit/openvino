// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <string>
#include <vector>
#include <xbyak_riscv/xbyak_riscv.hpp>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#ifdef SNIPPETS_DEBUG_CAPS

#    include <utility>

#    include "emitters/snippets/common/jit_segfault_detector_emitter_base.hpp"
#    include "jit_segfault_detector_emitter.hpp"
#    include "utils.hpp"

namespace ov::intel_cpu::riscv64 {

jit_uni_segfault_detector_emitter::jit_uni_segfault_detector_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                                                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                                                     jit_emitter* target_emitter,
                                                                     bool is_load,
                                                                     bool is_store,
                                                                     std::string target_node_name)
    : base_t(host, host_isa, target_emitter, is_load, is_store, std::move(target_node_name)) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

size_t jit_uni_segfault_detector_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_uni_segfault_detector_emitter::aux_gprs_count() const {
    return 2;
}

const jit_emitter* jit_uni_segfault_detector_emitter::get_target_emitter() const {
    return base_t::get_target_emitter();
}

void jit_uni_segfault_detector_emitter::save_target_emitter() const {
    std::vector<size_t> exclude_vec_regs(get_max_vecs_count());
    for (size_t i = 0; i < exclude_vec_regs.size(); ++i) {
        exclude_vec_regs[i] = i;
    }
    call_preamble({}, {}, exclude_vec_regs);

    const auto& set_local_handler_overload =
        static_cast<void (*)(jit_uni_segfault_detector_emitter*)>(set_local_handler);
    auto func_reg = Xbyak_riscv::Reg(aux_gpr_idxs[0]);
    h->uni_li(func_reg, reinterpret_cast<size_t>(set_local_handler_overload));
    h->uni_li(Xbyak_riscv::a0, reinterpret_cast<size_t>(this));
    h->jalr(Xbyak_riscv::ra, func_reg);

    call_postamble({}, {}, exclude_vec_regs);
}

void jit_uni_segfault_detector_emitter::set_local_handler(jit_uni_segfault_detector_emitter* emitter_address) {
    base_t::set_local_handler_impl(ov::intel_cpu::g_custom_segfault_handler<jit_uni_segfault_detector_emitter>,
                                   emitter_address);
}

void jit_uni_segfault_detector_emitter::memory_track(size_t gpr_idx_for_mem_address) const {
    Xbyak_riscv::Label label_set_address_current;
    Xbyak_riscv::Label label_set_address_end;

    const auto mem_reg = Xbyak_riscv::Reg(static_cast<int>(gpr_idx_for_mem_address));
    std::vector<size_t> used = {static_cast<size_t>(mem_reg.getIdx())};
    auto pool = aux_gpr_idxs;
    ov::intel_cpu::riscv64::utils::jit_aux_gpr_holder tmp_addr_holder(h, pool, used);
    used.push_back(static_cast<size_t>(tmp_addr_holder.get_reg().getIdx()));
    ov::intel_cpu::riscv64::utils::jit_aux_gpr_holder tmp_val_holder(h, pool, used);

    auto tmp_addr = tmp_addr_holder.get_reg();
    auto tmp_val = tmp_val_holder.get_reg();

    h->uni_li(tmp_addr, reinterpret_cast<size_t>(&start_address));
    h->ld(tmp_val, tmp_addr, 0);
    h->bne(tmp_val, Xbyak_riscv::zero, label_set_address_current);

    h->sd(mem_reg, tmp_addr, 0);
    h->uni_li(tmp_addr, reinterpret_cast<size_t>(&current_address));
    h->sd(mem_reg, tmp_addr, 0);
    h->j_(label_set_address_end);

    h->L(label_set_address_current);
    {
        h->uni_li(tmp_addr, reinterpret_cast<size_t>(&current_address));
        h->sd(mem_reg, tmp_addr, 0);
    }
    h->L(label_set_address_end);

    h->uni_li(tmp_addr, reinterpret_cast<size_t>(&iteration));
    h->ld(tmp_val, tmp_addr, 0);
    h->addi(tmp_val, tmp_val, 0x01);
    h->sd(tmp_val, tmp_addr, 0);
}

}  // namespace ov::intel_cpu::riscv64

#endif
