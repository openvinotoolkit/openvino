// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/jit_generator.hpp>
#include "snippets/snippets_isa.hpp"
#include "snippets/generator.hpp"

#include "jit_emitter.hpp"

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_exp_injector {
public:
    static size_t get_aux_vecs_count();

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    static void emit_impl(
        dnnl::impl::cpu::aarch64::jit_generator* h,
        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
        const std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map,
        const ov::element::Type exec_prc,
        const std::vector<size_t> &in_vec_idxs,
        const std::vector<size_t> &aux_vec_idxs,
        const std::vector<size_t> &out_vec_idxs,
        const Xbyak_aarch64::XReg& p_table);

    static void push_entry_map(std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map);
};

class jit_sigmoid_injector {
public:
    static size_t get_aux_vecs_count();

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    static void emit_impl(
            dnnl::impl::cpu::aarch64::jit_generator* h,
            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
            const std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map,
            const ov::element::Type exec_prc,
            const std::vector<size_t> &in_vec_idxs,
            const std::vector<size_t> &aux_vec_idxs,
            const std::vector<size_t> &out_vec_idxs,
            const Xbyak_aarch64::XReg& p_table);

    static void push_entry_map(std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map);
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
