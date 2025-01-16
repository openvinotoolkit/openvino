// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>

#include <cpu/aarch64/jit_generator.hpp>
#include "snippets/snippets_isa.hpp"
#include "snippets/generator.hpp"
#include "node.h"


namespace ov {
namespace intel_cpu {
namespace aarch64 {

enum emitter_in_out_map {
    vec_to_vec,
    vec_to_gpr,
    gpr_to_vec,
    gpr_to_gpr,
};

class jit_emitter : public ov::snippets::Emitter {
public:
    jit_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                ov::element::Type exec_prc = ov::element::f32,
                emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_vec) :
                Emitter(), h(host), host_isa_(host_isa), exec_prc_(exec_prc),
                in_out_type_(in_out_type), p_table(0), l_table (new Xbyak_aarch64::Label()) {
    }

    jit_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                const std::shared_ptr<ov::Node>& n,
                ov::element::Type exec_prc = ov::element::f32,
                emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_vec) :
                Emitter(), h(host), host_isa_(host_isa), exec_prc_(exec_prc),
                in_out_type_(in_out_type), p_table(0), l_table (new Xbyak_aarch64::Label()) {
    }

    void emit_code(
        const std::vector<size_t> &in_idxs,
        const std::vector<size_t> &out_idxs,
        const std::vector<size_t> &pool_vec_idxs = {},
        const std::vector<size_t> &pool_gpr_idxs = {}) const override;

    void emit_data() const override;

    virtual size_t get_inputs_count() const = 0;
    virtual size_t get_aux_vecs_count() const;
    virtual size_t get_aux_gprs_count() const;

    /**
     * @brief Returns supported precisions.
     * Precisions are ordered, the first bigger bitness precision with the same type will be selected.
     * Empty collection means the emitter supports any input precisions.
     */
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    size_t get_max_vecs_count() const;
    int32_t get_vec_length() const;

    mutable std::vector<size_t> aux_vec_idxs;
    mutable std::vector<size_t> aux_gpr_idxs;

    dnnl::impl::cpu::aarch64::jit_generator* h;
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa_;
    ov::element::Type exec_prc_;

    emitter_in_out_map in_out_type_;

    virtual void prepare_table();
    virtual void register_table_entries() {}

    void load_table_addr() const { h->adr(p_table, *l_table.get()); }

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

    mutable Xbyak_aarch64::XReg p_table;
    mutable std::shared_ptr<Xbyak_aarch64::Label> l_table;

    virtual void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const = 0;

    virtual void emitter_preamble(const std::vector<size_t>& in_idxs,
                                  const std::vector<size_t>& out_idxs,
                                  const std::vector<size_t>& pool_aux_vec_idxs,
                                  const std::vector<size_t>& pool_aux_gpr_idxs) const;

    virtual void emitter_postamble() const;

    void store_context(const std::unordered_set<size_t>& ignore_registers) const;

    void restore_context(const std::unordered_set<size_t>& ignore_registers) const;

    using table_t = std::multimap<std::string, table_entry_t>;
    using mapped_table_t = std::multimap<std::string, mapped_table_entry_t>;

    mapped_table_t entry_map_;

    Xbyak_aarch64::AdrImm table_val(std::string key, size_t key_off_val_shift = 0) const {
        const int32_t off = table_off(key, key_off_val_shift);
        return Xbyak_aarch64::ptr(p_table, off);
    }

    Xbyak_aarch64::AdrNoOfs table_val2(std::string key, size_t key_off_val_shift = 0) const {
        const int32_t off = table_off(key, key_off_val_shift);
        h->add_imm(h->X_DEFAULT_ADDR, p_table, off, h->X_TMP_0);
        return Xbyak_aarch64::ptr(h->X_DEFAULT_ADDR);
    }

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

private:
    mutable std::vector<size_t> preserved_vec_idxs;
    mutable std::vector<size_t> preserved_gpr_idxs;

    // General-purpose Registers
    static const std::vector<size_t> store_gpr_regs;

    size_t table_off(const std::string& key, const size_t key_off_val_shift = 0) const {
        // assumption: all table entries sharing the same key also
        // share their broadcast property
        const auto it = entry_map_.find(key); // search an entry for a key
        assert(it != entry_map_.end());
        const auto &te = (*it).second;
        const auto scale = te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }

    virtual void validate_arguments(const std::vector<size_t>&, const std::vector<size_t>&) const {}

    static inline size_t get_asimd_vectors_count() {
        return 32;
    }

    inline int32_t get_gpr_length() const {
        return h->x0.getBit() / 8;
    }

    void store_context(const std::vector<size_t>& gpr_regs,
                       const std::vector<size_t>& vec_regs,
                       const std::unordered_set<size_t>& ignore_vec_regs = {}) const;

    void restore_context(const std::vector<size_t>& gpr_regs,
                         const std::vector<size_t>& vec_regs,
                         const std::unordered_set<size_t>& ignore_vec_regs = {}) const;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
