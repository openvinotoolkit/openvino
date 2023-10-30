// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cxxabi.h>
#include <ie_common.h>
#include <cpu/x64/jit_generator.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/generator.hpp"
#include <node.h>

#include "openvino/runtime/threading/thread_local.hpp"

#include <set>

using namespace ov::threading;

namespace ov {
namespace intel_cpu {

class jit_emitter;
extern std::shared_ptr<ThreadLocal<jit_emitter*>> g_debug_err_handler;

enum emitter_in_out_map {
    vec_to_vec,
    vec_to_gpr,
    gpr_to_vec,
    gpr_to_gpr,
};

// structure for storage of emitter parameters to hash in map
struct emitter_params {
    virtual size_t hash() const = 0;
};

class jit_emitter : public ov::snippets::Emitter {
public:
    jit_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32, emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_vec)
        : Emitter(), h(host), host_isa_(host_isa), exec_prc_(exec_prc), l_table (new Xbyak::Label()), in_out_type_(in_out_type) {
        k_mask = Xbyak::Opmask(1); // FIXME: in general case we need preserve k_mask state as well
    }

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;
    void emit_data() const override;

    virtual size_t get_inputs_num() const = 0;
    virtual size_t aux_vecs_count() const;
    emitter_in_out_map get_in_out_type() const;

    /**
     * @brief Returns supported precisions.
     * Precisions are ordered, the first bigger bitness precision with the same type will be selected.
     * Empty collection means the emitter supports any input precisions.
     */
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

    virtual void print_debug_info() const {
        std::cerr << "Emitter type name:" << get_type_name(this)
            << " This is default info from base jit_emitter. Exact emitter type name is not set." << std::endl;
    }

protected:
    virtual size_t aux_gprs_count() const;

    size_t get_max_vecs_count() const;
    size_t get_vec_length() const;

    dnnl::impl::cpu::x64::jit_generator* h;
    dnnl::impl::cpu::x64::cpu_isa_t host_isa_;
    InferenceEngine::Precision exec_prc_;
    Xbyak::Opmask k_mask;

    virtual void prepare_table();
    virtual void register_table_entries() {}

    void load_table_addr() const { h->mov(p_table, *l_table.get()); }

    // we accept only 32bit hexadecimal table values to avoid any rounding
    using table_entry_val_t = uint32_t;
    using table_entry_offset_t = size_t; // offsets are in bytes wrt p_table
    using table_entry_bcast_t = bool; // true => bcast value

    struct table_entry_t {
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };
    struct mapped_table_entry_t {
        table_entry_offset_t off;
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };

    mutable Xbyak::Reg64 p_table;
    mutable std::shared_ptr<Xbyak::Label> l_table;

    enum {
        _cmp_eq_oq = dnnl::impl::cpu::x64::jit_generator::_cmp_eq_oq,
        _cmp_neq_uq = dnnl::impl::cpu::x64::jit_generator::_cmp_neq_uq,
        _cmp_lt_os = dnnl::impl::cpu::x64::jit_generator::_cmp_lt_os,
        _cmp_le_os = dnnl::impl::cpu::x64::jit_generator::_cmp_le_os,
        _cmp_ge_os = dnnl::impl::cpu::x64::jit_generator::_cmp_nlt_us,
        _cmp_gt_os = dnnl::impl::cpu::x64::jit_generator::_cmp_nle_us,
    };

    virtual void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const = 0;

    virtual void emitter_preamble(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                          const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const;
    virtual void emitter_postamble() const;

    emitter_in_out_map in_out_type_;

    mutable std::vector<size_t> aux_vec_idxs;
    mutable std::vector<size_t> aux_gpr_idxs;

    static constexpr int k_mask_size = 8;
    static constexpr int k_mask_num = 8;
    static constexpr int gpr_size_ = 8;

    Xbyak::Address table_val(std::string key, size_t key_off_val_shift = 0) const {
        auto off = table_off(key, key_off_val_shift);
        return h->ptr[p_table + off];
    }

    using table_t = std::multimap<std::string, table_entry_t>;
    using mapped_table_t = std::multimap<std::string, mapped_table_entry_t>;

    mapped_table_t entry_map_;

    void push_arg_entry_of(const std::string key, const table_entry_val_t val, const bool broadcast) {
        mapped_table_entry_t te {0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    }

    void push_entries_of(const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second; // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    }

    void build_debug_info() const;
    static void set_local_handler(jit_emitter* emitter_address);

    std::string get_type_name(const jit_emitter* emitter) const {
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

    // todo: remove when perf count PR merged
    // below 4 functions must be inline funtions to avoid corrupted rsp by function call, so defined inside class declaration.
    void internal_call_preamble() const {
        // gprs
        int gpr_size = 8;
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                         h->rax, h->rbx, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // mask regs
        // need preserve based on cpu capability, instead of host isa.
        // in case there are possibilty that different isa emitters exist in one subgraph KernelEmitter from perf standpoint in the future.
        // e.g. other emitters isa is avx512, while this emitter isa is avx2, and internal call is used. Internal call may use avx512 and spoil k-reg.
        // do not care about platform w/ avx512_common but w/o avx512_core(knight landing), which is obsoleted.
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            h->sub(h->rsp, k_mask_num * k_mask_size);
            for (size_t i = 0; i < k_mask_num; ++i) {
                h->kmovq(h->ptr[h->rsp + i * k_mask_size], Xbyak::Opmask(static_cast<int>(i)));
            }
        }

        // vector regs
        // 1. Caller obligation to save vector registers as callee may use them.
        // 2. There is an implicit assumption that the host code uses the same
        // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
        // `vlen` should be replaced with `host_isa::vlen` and
        // `host_isa::vecs_count`.
        h->sub(h->rsp, get_max_vecs_count() * get_vec_length());
        for (size_t i = 0; i < get_max_vecs_count(); ++i) {
            push_vec(h->ptr[h->rsp + i * get_vec_length()], i);
        }
    }
    void internal_call_postamble() const {
        int gpr_size = 8;
        // restore vector registers
        for (int i = static_cast<int>(get_max_vecs_count()) - 1; i >= 0; --i) {
            pop_vec(static_cast<size_t>(i), h->ptr[h->rsp + i * get_vec_length()]);
        }
        h->add(h->rsp, (get_max_vecs_count()) * get_vec_length());

        // restore k reg
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            for (int i = k_mask_num - 1; i >= 0; --i) {
                h->kmovq(Xbyak::Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
            }
            h->add(h->rsp, k_mask_num * k_mask_size);
        }

        // restore gpr registers
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                         h->rax, h->rbx, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);
        for (int i = n_gprs_to_save - 1; i >= 0; --i)
            h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
        h->add(h->rsp, n_gprs_to_save * gpr_size);
    }
    // align stack on 16-byte as ABI reqiures
    // callee is responsible to save and restore rbx. rbx must not be changed after call callee.
    void internal_call_rsp_align() const {
        h->mov(h->rbx, h->rsp);
        h->and_(h->rbx, 0xf);
        h->sub(h->rsp, h->rbx);
    }
    void internal_call_rsp_restore() const {
        h->add(h->rsp, h->rbx);
    }

private:
    mutable std::vector<size_t> preserved_vec_idxs;
    mutable std::vector<size_t> preserved_gpr_idxs;

    void push_vec(const Xbyak::Address &addr, size_t vec_idx) const;
    void pop_vec(size_t vec_idx, const Xbyak::Address &addr) const;

    size_t table_off(std::string& key, size_t key_off_val_shift = 0) const {
        // assumption: all table entries sharing the same key also
        // share their broadcast property
        // TODO: enforce through data structure
        const auto it = entry_map_.find(key); // search an entry for a key
        assert(it != entry_map_.end());
        const auto &te = (*it).second;
        const auto scale = te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }
    virtual void validate_arguments(const std::vector<size_t>&, const std::vector<size_t>&) const {}
};

}   // namespace intel_cpu
}   // namespace ov
