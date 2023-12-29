// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#include "jit_segfault_detector_emitter.hpp"

#include "jit_memory_emitters.hpp"
#include "jit_brgemm_emitter.hpp"
#include "jit_brgemm_copy_b_emitter.hpp"
#include "jit_kernel_emitter.hpp"

using namespace dnnl::impl::utils;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {

std::shared_ptr<ThreadLocal<jit_uni_segfault_detector_emitter*>> g_custom_segfault_handler =
    std::make_shared<ThreadLocal<jit_uni_segfault_detector_emitter*>>();

jit_uni_segfault_detector_emitter::jit_uni_segfault_detector_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
    jit_emitter* target_emitter, bool is_load, bool is_store, std::string target_node_name) :
    jit_emitter(host, host_isa),
    m_target_emitter(target_emitter),
    is_target_use_load_emitter(is_load),
    is_target_use_store_emitter(is_store),
    m_target_node_name(target_node_name) {
}

size_t jit_uni_segfault_detector_emitter::get_inputs_num() const { return 1; }

void jit_uni_segfault_detector_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    save_target_emitter();
    if (is_target_use_load_emitter) {
        memory_track(in_vec_idxs[0]);
    } else if (is_target_use_store_emitter) {
        memory_track(out_vec_idxs[0]);
    }
}

void jit_uni_segfault_detector_emitter::save_target_emitter() const {
    // use internal call as "->local" shoule be the execution thread. Otherwise always compilation thread.
    internal_call_preamble();

    const auto &set_local_handler_overload = static_cast<void (*)(jit_uni_segfault_detector_emitter*)>(set_local_handler);
    h->mov(h->rax, reinterpret_cast<size_t>(set_local_handler_overload));
    h->mov(abi_param1, reinterpret_cast<uint64_t>(this));
    internal_call_rsp_align();
    h->call(h->rax);
    internal_call_rsp_restore();

    internal_call_postamble();
}

void jit_uni_segfault_detector_emitter::set_local_handler(jit_uni_segfault_detector_emitter* emitter_address) {
    g_custom_segfault_handler->local() = emitter_address;
}

void jit_uni_segfault_detector_emitter::memory_track(size_t gpr_idx_for_mem_address) const {
    h->push(h->r15);
    Xbyak::Label label_set_address_current;
    Xbyak::Label label_set_address_end;
    h->mov(h->r15, reinterpret_cast<size_t>(&start_address));
    h->cmp(h->qword[h->r15], 0);
    h->jne(label_set_address_current);
    h->mov(h->qword[h->r15], Xbyak::Reg64(gpr_idx_for_mem_address));
    h->jmp(label_set_address_end);
    h->L(label_set_address_current);
    {
        h->mov(h->r15, reinterpret_cast<size_t>(&current_address));
        h->mov(h->qword[h->r15], Xbyak::Reg64(gpr_idx_for_mem_address));
    }
    h->L(label_set_address_end);
    // iteration++, 1 means first access
    h->mov(h->r15, reinterpret_cast<size_t>(&iteration));
    h->add(h->qword[h->r15], 0x01);
    h->pop(h->r15);
}

std::string jit_uni_segfault_detector_emitter::get_target_emitter_type_name(const jit_emitter* emitter) {
    std::string name = typeid(*emitter).name();
#ifndef _WIN32
    int status;
    std::unique_ptr<char, void (*)(void*)> demangled_name(
            abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status),
            std::free);
    name = demangled_name.get();
#endif
    return name;
}

template <typename T>
std::string join(const T& v, const std::string& sep = ", ") {
    std::ostringstream ss;
    size_t count = 0;
    for (const auto& x : v) {
        if (count++ > 0) {
            ss << sep;
        }
        ss << x;
    }
    return ss.str();
}

template <typename T>
std::string vector_to_string(const T& v) {
    std::ostringstream os;
    os << "[ " << ov::util::join(v) << " ]";
    return os.str();
}

void jit_uni_segfault_detector_emitter::print() {
    auto print_memory_emitter_info = [&](jit_memory_emitter *memory_emitter) {
        std::cerr << "detailed emitter info is, src precision:" << memory_emitter->src_prc << ", dst precision:" << memory_emitter->dst_prc
            << ", load/store element number:" << memory_emitter->count
            << ", byte offset" << memory_emitter->byte_offset << std::endl;
        // more memory address info tracked in detector_emitter.
        std::cerr << "start_address:" << start_address
            << ", current_address:" << current_address
            << ", iteration:" << iteration << "\n";
    };
    auto print_brgemm_emitter_info = [&](jit_brgemm_emitter* brgemm_emitter) {
        std::cerr << "detailed emitter info is, m_ctx.M:" << brgemm_emitter->m_ctx.M
            << " m_ctx.K:" << brgemm_emitter->m_ctx.K
            << " m_ctx.N:" << brgemm_emitter->m_ctx.N
            << " m_ctx.LDA:" << brgemm_emitter->m_ctx.LDA
            << " m_ctx.LDB:" << brgemm_emitter->m_ctx.LDB
            << " m_ctx.LDC:" << brgemm_emitter->m_ctx.LDC
            << " m_ctx.dt_in0:" << brgemm_emitter->m_ctx.dt_in0
            << " m_ctx.dt_in1:" << brgemm_emitter->m_ctx.dt_in1
            << " m_ctx.palette:" << brgemm_emitter->m_ctx.palette
            << " m_ctx.is_with_amx:" << brgemm_emitter->m_ctx.is_with_amx
            << " m_ctx.is_with_comp:" << brgemm_emitter->m_ctx.is_with_comp
            << " m_ctx.beta:" << brgemm_emitter->m_ctx.beta
            << " m_load_offset_a:" << brgemm_emitter->m_load_offset_a
            << " m_load_offset_b:" << brgemm_emitter->m_load_offset_b
            << " m_load_offset_scratch:" << brgemm_emitter->m_load_offset_scratch
            << " m_store_offset_c:" << brgemm_emitter->m_store_offset_c
            << " m_with_scratch:" << brgemm_emitter->m_with_scratch
            << " m_with_comp:" << brgemm_emitter->m_with_comp << "\n";
    };
    auto print_brgemm_copy_emitter_info = [&](jit_brgemm_copy_b_emitter* brgemm_copy_emitter) {
        std::cerr << "detailed emitter info is, m_LDB:" << brgemm_copy_emitter->m_LDB
            << " m_K:" << brgemm_copy_emitter->m_K
            << " m_K_blk:" << brgemm_copy_emitter->m_K_blk
            << " m_K_tail:" << brgemm_copy_emitter->m_K_tail
            << " m_N:" << brgemm_copy_emitter->m_N
            << " m_N_blk:" << brgemm_copy_emitter->m_N_blk
            << " m_N_tail:" << brgemm_copy_emitter->m_N_tail
            << " m_brgemm_prc_in0:" << brgemm_copy_emitter->m_brgemm_prc_in0
            << " m_brgemm_prc_in1:" << brgemm_copy_emitter->m_brgemm_prc_in1
            << " m_brgemmVNNIFactor:" << brgemm_copy_emitter->m_brgemmVNNIFactor
            << " m_with_comp:" << brgemm_copy_emitter->m_with_comp
            << " m_in_offset:" << brgemm_copy_emitter->m_in_offset
            << " m_out_offset:" << brgemm_copy_emitter->m_out_offset
            << " m_comp_offset:" << brgemm_copy_emitter->m_comp_offset << "\n";
    };
    auto print_kernel_emitter_info = [&](jit_kernel_emitter* kernel_emitter) {
        std::cerr << "detailed emitter info is, jcp.parallel_executor_ndims:"<< kernel_emitter->jcp.parallel_executor_ndims
            << " gp_regs_pool:"<< vector_to_string(kernel_emitter->gp_regs_pool)
            << " master_shape:" << vector_to_string(kernel_emitter->master_shape)
            << " num_inputs:" << kernel_emitter->num_inputs
            << " num_outputs:" << kernel_emitter->num_outputs
            << " num_unique_buffers:" << kernel_emitter->num_unique_buffers
            << " io_data_sizes:" << vector_to_string(kernel_emitter->io_data_sizes)
            << " data_ptr_regs_idx:" << vector_to_string(kernel_emitter->data_ptr_regs_idx)
            << " vec_regs_pool:" << vector_to_string(kernel_emitter->vec_regs_pool)
            << " reg_indexes_idx:" << kernel_emitter->reg_indexes_idx
            << " reg_const_params_idx:" << kernel_emitter->reg_const_params_idx << "\n";
        for (size_t i = 0; i < kernel_emitter->io_data_layouts.size(); ++i)
            std::cerr << "io_data_layouts for " << i << " is:"<< vector_to_string(kernel_emitter->io_data_layouts[i]) << "\n";
        for (size_t i = 0; i < kernel_emitter->io_shapes.size(); ++i)
            std::cerr << "io_shapes for " << i << " is:"<< vector_to_string(kernel_emitter->io_shapes[i]) << "\n";
    };

    std::cerr << "Node name:" << m_target_node_name << std::endl;
    std::cerr << "Emitter type name:" << get_target_emitter_type_name(m_target_emitter) << std::endl;
    if (auto *memory_emitter = dynamic_cast<jit_memory_emitter *>(m_target_emitter)) {
        print_memory_emitter_info(memory_emitter);
    } else if (auto *brgemm_emitter = dynamic_cast<jit_brgemm_emitter *>(m_target_emitter)) {
        print_brgemm_emitter_info(brgemm_emitter);
    } else if (auto *brgemm_copy_emitter = dynamic_cast<jit_brgemm_copy_b_emitter *>(m_target_emitter)) {
        print_brgemm_copy_emitter_info(brgemm_copy_emitter);
    } else if (auto *kernel_emitter = dynamic_cast<jit_kernel_emitter *>(m_target_emitter)) {
        print_kernel_emitter_info(kernel_emitter);
    }
}

}   // namespace intel_cpu
}   // namespace ov

#endif
